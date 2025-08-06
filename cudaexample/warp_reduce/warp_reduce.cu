#include<stdio.h>
#include<cuda_runtime.h>
#include<iostream>

// 对一维数组 input[] 进行规约（如求和），每个 block 得到一个归约值，保存在 output[blockIdx.x] 中
//采用两级规约
//warp内规约，每个warp得到一个规约值，使用warpReduceSum
//warp间规约，warp的结果存在Smem，然后由一个warp完成规约

__inline__ __device__
float warpReduceSum(float val){
    for(int offset=16;offset>0;offset>>=1)
        val+=__shfl_down_sync(0xffffffff,val,offset);
    return val;
}

__global__ void block_reduce_kernel(const float* input,float* output,int N){
    __shared__ float shared[32];
    int tid= threadIdx.x+blockIdx.x*blockDim.x;
    int lane=threadIdx.x%32;  //lane记录当前线程在warp内的编号
    int wid=threadIdx.x/32;   //wid记录当前线程在block中的第几个warp
    
    float val=(tid<N)?input[tid]:0.0f;
    val=warpReduceSum(val); //warp内规约

    if(lane==0)
        shared[wid]=val;
    __syncthreads();    //确保Smem中所有warp内的规约值写好，由一个warp来处理

    val=(threadIdx.x<32)?shared[lane]:0.0f;
    if(wid==0){
        val=warpReduceSum(val);
        if(lane==0)
            output[blockIdx.x]=val;
    }
}

#define N 1000
#define BLOCK_SIZE 256
int main(){
    float* h_input= new float[N];
    float* h_sums=new float[(N+BLOCK_SIZE-1)/BLOCK_SIZE];

    for(int i=0;i<N;i++)
        h_input[i]=1.0f;

    float *d_input,*d_sum;
    cudaMalloc(&d_input,N*sizeof(float));
    cudaMalloc(&d_sum,((N+BLOCK_SIZE-1)/BLOCK_SIZE)*sizeof(float));

    cudaMemcpy(d_input,h_input,N*sizeof(float),cudaMemcpyHostToDevice);
    int grid_size=(N+BLOCK_SIZE-1)/BLOCK_SIZE;
    block_reduce_kernel<<<grid_size,BLOCK_SIZE>>>(d_input,d_sum,N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_sums,d_sum,grid_size*sizeof(float),cudaMemcpyDeviceToHost);

    float total_sum=0.0f;
    for(int i=0;i<grid_size;i++)
       total_sum+=h_sums[i];

     std::cout<<"BLCOK SUM:"<<total_sum<<"\n";

    delete[] h_input;
    delete[] h_sums;
    cudaFree(d_input);
    cudaFree(d_sum);

}
#include<iostream>
#include<cuda_runtime.h>

//block内完成归约
//对输入数组 d_x 分块处理，每个 block 内部进行归约，然后将每个 block 的结果写入 d_y[blockIdx.x]
template<typename T>
__global__ void reduce_global(T* d_x,T* d_y)
{
    int tid=threadIdx.x;
    T* x=d_x+blockDim.x*blockIdx.x;
    for(int offset=blockDim.x>>1;offset>0;offset>>=1){
        if(tid<offset)
            x[tid]+=x[tid+offset];
        __syncthreads();
    }
    if(tid==0)
        d_y[blockIdx.x]=x[0];
}

int main() {
    using T = float;
    const int N = 1024;             // 总元素个数
    const int BLOCK_SIZE = 256;     // 每个 block 的线程数
    const int NUM_BLOCKS = N / BLOCK_SIZE; // 总 block 数（假设能整除）

    // 主机端数据
    T* h_x = new T[N];
    T* h_y = new T[NUM_BLOCKS];  // 每个 block 的归约结果

    // 初始化数据
    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.0f;  // 每个元素为 1，最后每个 block 的和应该是 256
    }

    // 分配设备内存
    T *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(T));
    cudaMalloc(&d_y, NUM_BLOCKS * sizeof(T));

    // 拷贝数据到设备
    cudaMemcpy(d_x, h_x, N * sizeof(T), cudaMemcpyHostToDevice);

    // 启动 kernel
    reduce_global<float><<<NUM_BLOCKS,BLOCK_SIZE>>>(d_x,d_y);
    cudaDeviceSynchronize();

    // 拷贝结果回主机
    cudaMemcpy(h_y, d_y, NUM_BLOCKS * sizeof(T), cudaMemcpyDeviceToHost);

    // 打印每个 block 的归约结果
    std::cout << "Block-wise reduce results:\n";
    for (int i = 0; i < NUM_BLOCKS; ++i) {
        std::cout << "Block " << i << ": " << h_y[i] << "\n";
    }

    // 最终汇总所有 block 的结果（可选）
    float total_sum = 0.0f;
    for (int i = 0; i < NUM_BLOCKS; ++i) {
        total_sum += h_y[i];
    }
    std::cout << "Total sum = " << total_sum << std::endl;

    // 清理资源
    delete[] h_x;
    delete[] h_y;
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
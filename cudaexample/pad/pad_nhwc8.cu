#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// 预定义快速除法结构，简化处理，这里直接用整除模
struct DivModFast {
    int divisor;
    __host__ __device__ DivModFast(int d) : divisor(d) {}
    __device__ __forceinline__ void divmod(int64_t val, int& div, int& mod) const {
        div = val / divisor;
        mod = val % divisor;
    }
};

// 简单pad索引计算，目前只支持常数pad
template <int MODE>
__device__ int pad_calc_in_idx_nhwc8(int out_idx, int pad, int input_dim, bool& use_pad_value) {
    if (MODE == 0) { // constant pad
        int in_idx = out_idx - pad;
        if (in_idx < 0 || in_idx >= input_dim) {
            use_pad_value = true;
            return 0;
        }
        return in_idx;
    }
    // 其他pad模式可扩展
    return 0;
}

template <typename T, int MODE>
__global__ void ppl_cukernel_pad_fast3_nhwc8(
    int64_t num_elems,              // N * H_out * W_out
    int num_dims,                   // = 3 (N,H,W)
    int vec_factor,                 // = 8 通道向量化长度
    int vec_dim_size,               // C/vec_factor (这里是1)
    float constant_value,           // 填充值
    const int64_t* input_dims,      // 输入尺寸[N,H_in,W]
    const int64_t* pads,            // pads 8维[N0,C0,H0,W0,N1,C1,H1,W1]
    const float4* input,            // 输入数据，vector化float4 (8个float16)
    float4* output)                 // 输出数据
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;

    bool use_pad = false;

    // 解析输出的 N, H_out, W_out 索引
    int64_t N = input_dims[0];
    int64_t H_in = input_dims[1];
    int64_t W = input_dims[2];

    // 计算输出的 H_out 从 pads 推断
    int64_t H_out = H_in + pads[6] + pads[2]; // H_out = H_in + H_before + H_after

    int out_n = index / (H_out * W);
    int remain = index % (H_out * W);
    int out_h = remain / W;
    int out_w = remain % W;

    // 计算输入索引，考虑pad
    int in_n = pad_calc_in_idx_nhwc8<MODE>(out_n, pads[0], N, use_pad);
    int in_h = pad_calc_in_idx_nhwc8<MODE>(out_h, pads[2], H_in, use_pad);
    int in_w = pad_calc_in_idx_nhwc8<MODE>(out_w, pads[3], W, use_pad);

    int64_t input_offset = ((in_n * H_in + in_h) * W + in_w) * vec_dim_size;
    int64_t output_offset = index * vec_dim_size;

    // 填充值vector
    float4 pad_val;
    T* pad_ptr = reinterpret_cast<T*>(&pad_val);
    for (int i = 0; i < vec_factor; ++i) {
        pad_ptr[i] = (T)constant_value;
    }

    for (int i = 0; i < vec_dim_size; ++i) {
        output[output_offset + i] = use_pad ? pad_val : input[input_offset + i];
    }
}

// 辅助函数打印float16数据（NHWC8）
void print_output(const __half* data, int N, int H, int W, int C) {
    std::cout << "Output (NHWC8):\n";
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                std::cout << "[";
                for (int c = 0; c < C; ++c) {
                    int idx = ((n * H + h) * W + w) * C + c;
                    std::cout << __half2float(data[idx]);
                    if (c != C - 1) std::cout << ",";
                }
                std::cout << "] ";
            }
            std::cout << "\n";
        }
        std::cout << "-----\n";
    }
}

int main() {
    using T = __half;
    const int N = 1, H_in = 2, W = 3, C_in = 2;
    const int C_padded = 8; // NHWC8格式C=8
    const int H_after_pad = 2;
    const int H_out = H_in + H_after_pad;

    // 输入元素总数
    int64_t input_elems = int64_t(N) * H_in * W * C_padded;
    int64_t output_elems = int64_t(N) * H_out * W * C_padded;

    // 申请host内存
    std::vector<T> h_input(input_elems, __float2half(0));
    std::vector<T> h_output(output_elems, __float2half(-1)); // 初始化为-1方便观察

    // 初始化输入，填充前C=2通道有效，后6通道填0
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H_in; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int c = 0; c < C_in; ++c) {
                    int idx = ((n * H_in + h) * W + w) * C_padded + c;
                    h_input[idx] = __float2half(float(idx)); // 简单填值，方便验证
                }
                // 后6通道默认是0（已初始化）
            }
        }
    }

    // CUDA malloc
    T *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, sizeof(T) * input_elems);
    cudaMalloc(&d_output, sizeof(T) * output_elems);

    cudaMemcpy(d_input, h_input.data(), sizeof(T) * input_elems, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output.data(), sizeof(T) * output_elems, cudaMemcpyHostToDevice);

    // dims 和 pads
    int64_t h_input_dims[3] = {N, H_in, W};
    int64_t h_pads[8] = {0, 0, 0, 0, 0, 0, H_after_pad, 0};

    int64_t* d_input_dims;
    int64_t* d_pads;
    cudaMalloc(&d_input_dims, sizeof(int64_t) * 3);
    cudaMalloc(&d_pads, sizeof(int64_t) * 8);
    cudaMemcpy(d_input_dims, h_input_dims, sizeof(int64_t) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pads, h_pads, sizeof(int64_t) * 8, cudaMemcpyHostToDevice);

    // kernel参数
    int64_t num_elems = int64_t(N) * H_out * W; // N * H_out * W_out
    int vec_factor = 8;      // float4 vector: 8 x half
    int vec_dim_size = C_padded / vec_factor; // 1

    dim3 block(256);
    dim3 grid((num_elems + block.x -1)/block.x);

    ppl_cukernel_pad_fast3_nhwc8<T, 0><<<grid, block>>>(
        num_elems,
        3,
        vec_factor,
        vec_dim_size,
        0.0f,       // pad常数值0
        d_input_dims,
        d_pads,
        (const float4*)d_input,
        (float4*)d_output);

    cudaDeviceSynchronize();

    // 拷贝输出回host
    cudaMemcpy(h_output.data(), d_output, sizeof(T) * output_elems, cudaMemcpyDeviceToHost);

    // 打印输入与输出
    std::cout << "Input tensor (NHWC8, N=1, H=2, W=3, C=8):\n";
    print_output(h_input.data(), N, H_in, W, C_padded);

    std::cout << "\nOutput tensor after pad (NHWC8, N=1, H=4, W=3, C=8):\n";
    print_output(h_output.data(), N, H_out, W, C_padded);

    // 释放内存
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_input_dims);
    cudaFree(d_pads);

    return 0;
}

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cmath>

__global__ void pad_nhwc8_fp16_kernel(
    int64_t num_elems,        // N * out_H * out_W
    int in_N, int in_H, int in_W, int in_C,
    int out_H, int out_W, int out_C,
    int pad_h_beg,
    half constant_val,
    const half* __restrict__ input,
    half* __restrict__ output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elems) return;

    int n = idx / (out_H * out_W);
    int remain = idx % (out_H * out_W);
    int h = remain / out_W;
    int w = remain % out_W;

    int in_C8 = (in_C + 7) / 8 * 8;
    int out_C8 = (out_C + 7) / 8 * 8;

    int in_h = h - pad_h_beg;

    bool h_in_range = (in_h >= 0 && in_h < in_H);

    for (int c = 0; c < out_C; ++c) {
        int c_outer = c / 8;
        int c_inner = c % 8;
        int out_offset = (((n * out_H + h) * out_W + w) * (out_C8 / 8) + c_outer) * 8 + c_inner;

        if (!h_in_range || c >= in_C) {
            output[out_offset] = constant_val;
        } else {
            int in_offset = (((n * in_H + in_h) * in_W + w) * (in_C8 / 8) + c_outer) * 8 + c_inner;
            output[out_offset] = input[in_offset];
        }
    }

    // 填充对齐部分 (c >= out_C, 但 < out_C8)
    for (int c = out_C; c < out_C8; ++c) {
        int c_outer = c / 8;
        int c_inner = c % 8;
        int out_offset = (((n * out_H + h) * out_W + w) * (out_C8 / 8) + c_outer) * 8 + c_inner;
        output[out_offset] = constant_val;
    }
}
int main() {
    int N = 1, H = 2, W = 1, C = 5;
    int pad_h_end = 2, pad_h_beg = 0;
    int out_H = H + pad_h_beg + pad_h_end;
    int out_C = C;

    int in_C8 = (C + 7) / 8 * 8;
    int out_C8 = (out_C + 7) / 8 * 8;

    int64_t in_elems = N * H * W * in_C8;
    int64_t out_elems = N * out_H * W;

    half* h_input = new half[in_elems];
    for (int i = 0; i < in_elems; ++i) {
        h_input[i] = __float2half(i);  // 全部设为 1.0f
    }

    half* d_input;
    half* d_output;
    cudaMalloc(&d_input, in_elems * sizeof(half));
    cudaMalloc(&d_output, N * out_H * W * out_C8 * sizeof(half));
    cudaMemcpy(d_input, h_input, in_elems * sizeof(half), cudaMemcpyHostToDevice);

    half pad_val = __float2half(0.0f);
    int block = 64;
    int grid = (out_elems + block - 1) / block;

    pad_nhwc8_fp16_kernel<<<grid, block>>>(
        out_elems, N, H, W, C,
        out_H, W, out_C,
        pad_h_beg, pad_val,
        d_input, d_output);

    half* h_output = new half[N * out_H * W * out_C8];
    cudaMemcpy(h_output, d_output, N * out_H * W * out_C8 * sizeof(half), cudaMemcpyDeviceToHost);

    // 展示输出数据
    for (int i = 0; i < N * out_H * W * out_C8; ++i) {
        std::cout << __half2float(h_output[i]) << " ";
        if ((i + 1) % out_C8 == 0) std::cout << "\n";
    }

    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

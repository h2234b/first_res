#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Kernel definition: pad_nhwc8_fp16_constant
__global__ void pad_nhwc8_fp16_constant_kernel(
    int64_t num_elems,       // N * H * W
    int in_N, int in_H, int in_W, int in_C,
    int out_H, int out_W, int out_C,
    int pad_n_beg, int pad_h_beg, int pad_w_beg, int pad_c_beg,
    half constant_val,
    const half* __restrict__ input,
    half* __restrict__ output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;

    int N = index / (out_H * out_W);
    int remain = index % (out_H * out_W);
    int H = remain / out_W;
    int W = remain % out_W;

    int in_C_padded = (in_C + 7) / 8 * 8;
    int out_C_padded = (out_C + 7) / 8 * 8;

    for (int C = 0; C < out_C_padded; ++C) {
        int in_n = N - pad_n_beg;
        int in_h = H - pad_h_beg;
        int in_w = W - pad_w_beg;
        int in_c = C - pad_c_beg;

        bool is_pad = (in_n < 0 || in_n >= in_N ||
                       in_h < 0 || in_h >= in_H ||
                       in_w < 0 || in_w >= in_W ||
                       in_c < 0 || in_c >= in_C);

        int out_offset = ((N * out_H + H) * out_W + W) * out_C_padded + C;

        if (is_pad) {
            output[out_offset] = constant_val;
        } else {
            int in_offset = ((in_n * in_H + in_h) * in_W + in_w) * in_C_padded + in_c;
            output[out_offset] = input[in_offset];
        }
    }
}

// CUDA kernel launch code
int main() {
    // Define input dimensions and padding values
    int N = 1, H = 2, W = 1, C = 5;  // Input size (N, H, W, C)
    int pad_n_beg = 0, pad_h_beg = 0, pad_w_beg = 0, pad_c_beg = 0;
    int pad_n_end = 0, pad_h_end = 2, pad_w_end = 0, pad_c_end = 0;

    int out_H = H + pad_h_beg + pad_h_end;
    int out_W = W + pad_w_beg + pad_w_end;
    int out_C = C + pad_c_beg + pad_c_end;
    int64_t num_elems = N * out_H * out_W;

    // Allocate memory for input and output
    half* d_input;
    half* d_output;
    cudaMalloc(&d_input, N * H * W * C * sizeof(half));
    cudaMalloc(&d_output, N * out_H * out_W * out_C * sizeof(half));

    // Initialize input with random values (for testing purposes)
    half* h_input = new half[N * H * W * C];
    for (int i = 0; i < N * H * W * C; ++i) {
        h_input[i] = __float2half(1.0f);  // Initialize all input values to 1.0f
    }
    cudaMemcpy(d_input, h_input, N * H * W * C * sizeof(half), cudaMemcpyHostToDevice);

    // Launch kernel
    half constant_val = __float2half(0.0f);  // Pad value (constant padding)

    int block_size = 256;
    int grid_size = (num_elems + block_size - 1) / block_size;

    pad_nhwc8_fp16_constant_kernel<<<grid_size, block_size>>>(num_elems, N, H, W, C,
        out_H, out_W, out_C, pad_n_beg, pad_h_beg, pad_w_beg, pad_c_beg, constant_val, d_input, d_output);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy output back to host for verification
    half* h_output = new half[N * out_H * out_W * out_C];
    cudaMemcpy(h_output, d_output, N * out_H * out_W * out_C * sizeof(half), cudaMemcpyDeviceToHost);

    // Verification output
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < out_H; ++h) {
            for (int w = 0; w < out_W; ++w) {
                for (int c = 0; c < out_C; ++c) {
                    int idx = ((n * out_H + h) * out_W + w) * out_C + c;
                    std::cout << "output[" << n << "," << h << "," << w << "," << c << "] = " 
                              << __half2float(h_output[idx]) << std::endl;
                }
            }
        }
    }

    // Free allocated memory
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

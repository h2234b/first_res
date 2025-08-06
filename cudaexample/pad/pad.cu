#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cassert>

template <typename T>
struct GArray {
    T* ptr;
    __host__ __device__ T& operator[](int i) const { return ptr[i]; }
};

struct DivModFast {
    int divisor;
    __host__ __device__ DivModFast() {}
    __host__ __device__ DivModFast(int d) : divisor(d) {}
    __host__ __device__ void divmod(int value, int& div, int& mod) const {
        div = value / divisor;
        mod = value % divisor;
    }
};

template <int MODE>
__device__ int pad_calc_in_idx(int out_idx, int pad, int input_dim, bool& use_pad_value) {
    if (MODE == 0) { // constant
        int in_idx = out_idx - pad;
        if (in_idx < 0 || in_idx >= input_dim) {
            use_pad_value = true;
            return 0;
        }
        return in_idx;
    }
    return 0;
}

template <typename T, int MODE>
__global__ void ppl_cukernel_pad_fast3(
    int64_t num_elems,                // N*H*W
    int num_dims,                     // =3 (N,H,W)
    int vec_factor,                   // =1 (float16, no vectorization)
    int vec_dim_size,                 // =C
    float constant_value,
    GArray<int64_t> input_dims,       // [N, H, W]
    GArray<int64_t> input_strides,    // [H*W, W, 1]
    const T* input,
    const int64_t* pads,              // [N_pad0,C_pad0,H_pad0,W_pad0,N_pad1,C_pad1,H_pad1,W_pad1]
    GArray<DivModFast> output_strides_fast, // [H*W, W, 1]
    T* output)
{
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;

    bool use_pad_value = false;
    T constant_val = (T)constant_value;

    int64_t input_offset = 0;
    int out_idx, remain = index;

    for (int it = 0; (it < num_dims) && !use_pad_value; ++it) {
        output_strides_fast[it].divmod(remain, out_idx, remain);
        int64_t start_pad_val = pads[it];
        int in_idx = pad_calc_in_idx<MODE>(out_idx, start_pad_val, input_dims[it], use_pad_value);
        input_offset += in_idx * input_strides[it];
    }

    for (int i = 0; i < vec_dim_size; ++i) {
        int out_offset = index * vec_dim_size + i;
        int in_offset = input_offset * vec_dim_size + i;
        output[out_offset] = use_pad_value ? constant_val : input[in_offset];
    }
}
int main() {
    using T = __half;
    int N = 1, C = 2, H = 3, W = 2;
    int pad_c_after = 2;
    int C_out = C + pad_c_after;

    int64_t num_elems = N * H * W;
    int vec_factor = 1;
    int vec_dim_size = C;

    int64_t total_in = num_elems * C;
    int64_t total_out = num_elems * C_out;

    std::vector<T> h_input(total_in);
    std::vector<T> h_output(total_out, __float2half(0.0f));

    // 填充输入：0 ~ total_in-1
    for (int i = 0; i < total_in; ++i) {
        h_input[i] = __float2half(float(i));
    }

    T* d_input;
    T* d_output;
    cudaMalloc(&d_input, sizeof(T) * total_in);
    cudaMalloc(&d_output, sizeof(T) * total_out);
    cudaMemcpy(d_input, h_input.data(), sizeof(T) * total_in, cudaMemcpyHostToDevice);

    // 形状与步长
    int64_t h_input_dims[3] = {N, H, W};
    int64_t h_input_strides[3] = {H * W, W, 1};
    DivModFast h_output_strides[3] = {DivModFast(H * W), DivModFast(W), DivModFast(1)};
    int64_t h_pads[8] = {0, 0, 0, 0, 0, pad_c_after, 0, 0};

    // device copy
    int64_t *d_input_dims, *d_input_strides, *d_pads;
    DivModFast *d_output_strides;
    cudaMalloc(&d_input_dims, sizeof(int64_t) * 3);
    cudaMalloc(&d_input_strides, sizeof(int64_t) * 3);
    cudaMalloc(&d_pads, sizeof(int64_t) * 8);
    cudaMalloc(&d_output_strides, sizeof(DivModFast) * 3);
    cudaMemcpy(d_input_dims, h_input_dims, sizeof(int64_t) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_strides, h_input_strides, sizeof(int64_t) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_strides, h_output_strides, sizeof(DivModFast) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pads, h_pads, sizeof(int64_t) * 8, cudaMemcpyHostToDevice);

    GArray<int64_t> g_input_dims{d_input_dims};
    GArray<int64_t> g_input_strides{d_input_strides};
    GArray<DivModFast> g_output_strides{d_output_strides};

    dim3 block(256);
    dim3 grid((num_elems + block.x - 1) / block.x);

    ppl_cukernel_pad_fast3<__half, 0><<<grid, block>>>(
        num_elems, 3, vec_factor, vec_dim_size, 99.0f,
        g_input_dims, g_input_strides,
        d_input, d_pads, g_output_strides, d_output);

    cudaMemcpy(h_output.data(), d_output, sizeof(T) * total_out, cudaMemcpyDeviceToHost);

    std::cout << "\n======= 原始输入 (NCHW) =======\n";
    for (int c = 0; c < C; ++c) {
        std::cout << "C=" << c << ":\n";
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int offset = ((0 * C + c) * H + h) * W + w;
                std::cout << __half2float(h_input[offset]) << " ";
            }
            std::cout << "\n";
        }
    }

    std::cout << "\n======= 输出 (padding 后, NCHW, C=4) =======\n";
    for (int c = 0; c < C_out; ++c) {
        std::cout << "C=" << c;
        if (c >= C) std::cout << " (pad)";
        std::cout << ":\n";
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int offset = ((0 * C_out + c) * H + h) * W + w;
                std::cout << __half2float(h_output[offset]) << " ";
            }
            std::cout << "\n";
        }
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_input_dims);
    cudaFree(d_input_strides);
    cudaFree(d_output_strides);
    cudaFree(d_pads);

    return 0;
}

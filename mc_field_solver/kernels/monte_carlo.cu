#include "monte_carlo.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) \
    throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e_)); } while (0)

__global__ void monte_carlo_kernel(const std::uint8_t* cell_type,
    const float* boundary_value, const int* interior_indices, int num_interior,
    int nx, int num_walks, unsigned long long seed, float* potential, float* standard_error) {
    const int p = blockIdx.x;
    if (p >= num_interior) return;
    const int tid = threadIdx.x;
    const int start = interior_indices[p];
    double local_sum = 0.0, local_sum2 = 0.0;
    curandStatePhilox4_32_10_t rng;
    const unsigned long long sequence =
        static_cast<unsigned long long>(p) * blockDim.x + tid;
    curand_init(seed, sequence, 0, &rng);
    for (int walk = tid; walk < num_walks; walk += blockDim.x) {
        int x = start % nx, y = start / nx;
        while (cell_type[y * nx + x] != static_cast<std::uint8_t>(CellType::Boundary)) {
            switch (curand(&rng) & 3U) {
                case 0: ++x; break;
                case 1: --x; break;
                case 2: ++y; break;
                default: --y; break;
            }
        }
        const double value = boundary_value[y * nx + x];
        local_sum += value;
        local_sum2 += value * value;
    }
    extern __shared__ double shared[];
    double* sums = shared;
    double* sums2 = shared + blockDim.x;
    sums[tid] = local_sum;
    sums2[tid] = local_sum2;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
            sums2[tid] += sums2[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const double mean = sums[0] / num_walks;
        const double variance = fmax(0.0, sums2[0] / num_walks - mean * mean);
        potential[start] = static_cast<float>(mean);
        standard_error[start] = static_cast<float>(sqrt(variance / num_walks));
    }
}

__global__ void gradient_kernel(const float* potential, float* ex, float* ey,
                                float* magnitude, int nx, int ny, float dx, float dy) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || x >= nx - 1 || y <= 0 || y >= ny - 1) return;
    const int idx = y * nx + x;
    const float field_x = -(potential[idx + 1] - potential[idx - 1]) / (2.0f * dx);
    const float field_y = -(potential[idx + nx] - potential[idx - nx]) / (2.0f * dy);
    ex[idx] = field_x;
    ey[idx] = field_y;
    magnitude[idx] = sqrtf(field_x * field_x + field_y * field_y);
}

template <typename T> static T* copy_to_device(const std::vector<T>& host) {
    T* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, host.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

Solution solve_cuda(const Domain& d, int num_walks, std::uint64_t seed, int block_size) {
    if (num_walks <= 0) throw std::invalid_argument("walks must be positive");
    if (block_size < 1 || block_size > 1024 || (block_size & (block_size - 1)))
        throw std::invalid_argument("block size must be a power of two in [1, 1024]");
    Solution out;
    const std::size_t count = static_cast<std::size_t>(d.nx) * d.ny;
    out.potential = d.boundary_value;
    out.standard_error.assign(count, 0.0f);
    std::uint8_t* d_type = copy_to_device(d.cell_type);
    float* d_boundary = copy_to_device(d.boundary_value);
    int* d_interior = copy_to_device(d.interior_indices);
    float *d_potential = nullptr, *d_stderr = nullptr;
    float *d_ex = nullptr, *d_ey = nullptr, *d_magnitude = nullptr;
    CUDA_CHECK(cudaMalloc(&d_potential, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_stderr, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ex, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ey, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_magnitude, count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_potential, out.potential.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_stderr, 0, count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ex, 0, count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ey, 0, count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_magnitude, 0, count * sizeof(float)));
    const auto begin = std::chrono::steady_clock::now();
    monte_carlo_kernel<<<static_cast<unsigned>(d.interior_indices.size()), block_size,
        2 * block_size * sizeof(double)>>>(d_type, d_boundary, d_interior,
        static_cast<int>(d.interior_indices.size()), d.nx, num_walks, seed,
        d_potential, d_stderr);
    CUDA_CHECK(cudaGetLastError());
    const dim3 threads(16, 16);
    const dim3 blocks((d.nx + threads.x - 1) / threads.x,
                      (d.ny + threads.y - 1) / threads.y);
    gradient_kernel<<<blocks, threads>>>(d_potential, d_ex, d_ey, d_magnitude,
                                         d.nx, d.ny, d.dx, d.dy);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    out.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    CUDA_CHECK(cudaMemcpy(out.potential.data(), d_potential, count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.standard_error.data(), d_stderr, count * sizeof(float), cudaMemcpyDeviceToHost));
    out.ex.resize(count); out.ey.resize(count); out.magnitude.resize(count);
    CUDA_CHECK(cudaMemcpy(out.ex.data(), d_ex, count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.ey.data(), d_ey, count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.magnitude.data(), d_magnitude, count * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_type); cudaFree(d_boundary); cudaFree(d_interior);
    cudaFree(d_potential); cudaFree(d_stderr);
    cudaFree(d_ex); cudaFree(d_ey); cudaFree(d_magnitude);
    return out;
}

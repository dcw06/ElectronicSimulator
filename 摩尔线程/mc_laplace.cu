#include "gpu_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#define GPU_CHECK(call)                                                         \
  do {                                                                          \
    gpuError_t error__ = (call);                                                 \
    if (error__ != GPU_SUCCESS) {                                                \
      std::fprintf(stderr, "%s error at %s:%d: %s\n", GPU_BACKEND_NAME,         \
                   __FILE__, __LINE__, gpuGetErrorString(error__));               \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

// One block estimates one grid point. Each thread generates several independent
// first-hit random walks, then the block reduces their payoffs in shared memory.
__global__ void solve_laplace_mc(const unsigned char* boundary,
                                 const float* boundary_value, float* potential,
                                 int nx, int ny, int walks_per_point,
                                 int max_steps, unsigned long long seed) {
  const int point = blockIdx.x;
  const int n = nx * ny;
  if (point >= n) return;

  if (boundary[point]) {
    if (threadIdx.x == 0) potential[point] = boundary_value[point];
    return;
  }

  gpuRngState rng;
  gpuRngInit(seed, static_cast<unsigned long long>(point) * blockDim.x +
                       threadIdx.x,
             0, &rng);

  float local_sum = 0.0f;
  int local_count = 0;
  for (int walk = threadIdx.x; walk < walks_per_point; walk += blockDim.x) {
    int x = point % nx;
    int y = point / nx;
    bool hit = false;
    for (int step = 0; step < max_steps; ++step) {
      const unsigned direction = gpuRngNext(&rng) & 3u;
      x += direction == 0u ? 1 : direction == 1u ? -1 : 0;
      y += direction == 2u ? 1 : direction == 3u ? -1 : 0;

      // The outer grid must be marked as boundary. This guard also prevents an
      // invalid access if a malformed mask is supplied.
      x = max(0, min(nx - 1, x));
      y = max(0, min(ny - 1, y));
      const int q = y * nx + x;
      if (boundary[q]) {
        local_sum += boundary_value[q];
        ++local_count;
        hit = true;
        break;
      }
    }
    // A walk exceeding max_steps is omitted rather than silently biased to zero.
    (void)hit;
  }

  extern __shared__ unsigned char shared_raw[];
  float* sums = reinterpret_cast<float*>(shared_raw);
  int* counts = reinterpret_cast<int*>(sums + blockDim.x);
  sums[threadIdx.x] = local_sum;
  counts[threadIdx.x] = local_count;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sums[threadIdx.x] += sums[threadIdx.x + stride];
      counts[threadIdx.x] += counts[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    potential[point] = counts[0] ? sums[0] / counts[0] : NAN;
}

__global__ void negative_gradient(const float* u, float* ex, float* ey,
                                  int nx, int ny, float dx, float dy) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= nx * ny) return;
  const int x = p % nx;
  const int y = p / nx;
  const int xl = max(0, x - 1), xr = min(nx - 1, x + 1);
  const int yd = max(0, y - 1), yu = min(ny - 1, y + 1);
  ex[p] = -(u[y * nx + xr] - u[y * nx + xl]) / ((xr - xl) * dx);
  ey[p] = -(u[yu * nx + x] - u[yd * nx + x]) / ((yu - yd) * dy);
}

static void write_csv(const char* path, const std::vector<float>& u,
                      const std::vector<float>& ex,
                      const std::vector<float>& ey, int nx, int ny) {
  std::ofstream out(path);
  out << "x,y,u,u_true,u_error,abs_u_error,Ex,Ey,Ex_true,Ey_true,"
         "E_error\n";
  for (int y = 0; y < ny; ++y)
    for (int x = 0; x < nx; ++x) {
      const int p = y * nx + x;
      const float px = static_cast<float>(x) / (nx - 1);
      const float py = static_cast<float>(y) / (ny - 1);
      const float u_true = px * px - py * py;
      const float ex_true = -2.0f * px;
      const float ey_true = 2.0f * py;
      const float u_error = u[p] - u_true;
      const float e_error =
          std::hypot(ex[p] - ex_true, ey[p] - ey_true);
      out << px << ',' << py << ',' << u[p] << ',' << u_true << ','
          << u_error << ',' << std::abs(u_error) << ',' << ex[p] << ','
          << ey[p] << ',' << ex_true << ',' << ey_true << ',' << e_error
          << '\n';
    }
}

int main(int argc, char** argv) {
  const int nx = argc > 1 ? std::max(3, std::atoi(argv[1])) : 128;
  const int ny = argc > 2 ? std::max(3, std::atoi(argv[2])) : 128;
  const int walks = argc > 3 ? std::max(1, std::atoi(argv[3])) : 4096;
  const int max_steps = argc > 4 ? std::max(1, std::atoi(argv[4])) : nx * ny * 8;
  const int n = nx * ny;

  std::vector<unsigned char> boundary(n, 0);
  std::vector<float> values(n, 0.0f);
  // Unit-square verification case: u(x,y)=x^2-y^2 is harmonic. Apply its
  // analytic value on all four edges as the Dirichlet boundary condition.
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      if (x != 0 && x != nx - 1 && y != 0 && y != ny - 1) continue;
      const float px = static_cast<float>(x) / (nx - 1);
      const float py = static_cast<float>(y) / (ny - 1);
      const int p = y * nx + x;
      boundary[p] = 1;
      values[p] = px * px - py * py;
    }
  }

  unsigned char* d_boundary = nullptr;
  float *d_values = nullptr, *d_u = nullptr, *d_ex = nullptr, *d_ey = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&d_boundary),
                      n * sizeof(*d_boundary)));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&d_values), n * sizeof(*d_values)));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&d_u), n * sizeof(*d_u)));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&d_ex), n * sizeof(*d_ex)));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&d_ey), n * sizeof(*d_ey)));
  GPU_CHECK(gpuMemcpy(d_boundary, boundary.data(), n * sizeof(*d_boundary),
                      gpuMemcpyHostToDevice));
  GPU_CHECK(gpuMemcpy(d_values, values.data(), n * sizeof(*d_values),
                      gpuMemcpyHostToDevice));

  constexpr int threads = 256;
  const size_t shared_bytes = threads * (sizeof(float) + sizeof(int));
  solve_laplace_mc<<<n, threads, shared_bytes>>>(
      d_boundary, d_values, d_u, nx, ny, walks, max_steps, 20260809ULL);
  GPU_CHECK(gpuGetLastError());
  negative_gradient<<<(n + threads - 1) / threads, threads>>>(
      d_u, d_ex, d_ey, nx, ny, 1.0f / (nx - 1), 1.0f / (ny - 1));
  GPU_CHECK(gpuGetLastError());
  GPU_CHECK(gpuDeviceSynchronize());

  std::vector<float> u(n), ex(n), ey(n);
  GPU_CHECK(gpuMemcpy(u.data(), d_u, n * sizeof(float), gpuMemcpyDeviceToHost));
  GPU_CHECK(gpuMemcpy(ex.data(), d_ex, n * sizeof(float), gpuMemcpyDeviceToHost));
  GPU_CHECK(gpuMemcpy(ey.data(), d_ey, n * sizeof(float), gpuMemcpyDeviceToHost));
  write_csv("field.csv", u, ex, ey, nx, ny);

  double squared_error_sum = 0.0;
  float max_abs_error = 0.0f;
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      const int p = y * nx + x;
      const float px = static_cast<float>(x) / (nx - 1);
      const float py = static_cast<float>(y) / (ny - 1);
      const float error = u[p] - (px * px - py * py);
      squared_error_sum += static_cast<double>(error) * error;
      max_abs_error = std::max(max_abs_error, std::abs(error));
    }
  }
  const double rmse = std::sqrt(squared_error_sum / n);
  std::cout << GPU_BACKEND_NAME << ": wrote field.csv (" << nx << " x " << ny
            << ", " << walks
            << " walks/point), potential RMSE=" << rmse
            << ", max_abs_error=" << max_abs_error << '\n';

  gpuFree(d_boundary);
  gpuFree(d_values);
  gpuFree(d_u);
  gpuFree(d_ex);
  gpuFree(d_ey);
}

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

static float exact_u(float x, float y) { return x * x - y * y; }

int main(int argc, char** argv) {
  const int nx = argc > 1 ? std::max(3, std::atoi(argv[1])) : 32;
  const int ny = argc > 2 ? std::max(3, std::atoi(argv[2])) : 32;
  const int walks = argc > 3 ? std::max(1, std::atoi(argv[3])) : 4096;
  const int max_steps = argc > 4 ? std::max(1, std::atoi(argv[4])) : nx * ny * 8;
  const int n = nx * ny;

  std::vector<unsigned char> boundary(n, 0);
  std::vector<float> values(n, 0.0f), u(n, 0.0f), ex(n), ey(n);
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      if (x != 0 && x != nx - 1 && y != 0 && y != ny - 1) continue;
      const int p = y * nx + x;
      boundary[p] = 1;
      values[p] = exact_u(static_cast<float>(x) / (nx - 1),
                          static_cast<float>(y) / (ny - 1));
      u[p] = values[p];
    }
  }

  std::mt19937 rng(20260809U);
  std::uniform_int_distribution<unsigned> direction(0, 3);
  long long truncated_walks = 0;
  for (int p = 0; p < n; ++p) {
    if (boundary[p]) continue;
    double sum = 0.0;
    int hits = 0;
    for (int walk = 0; walk < walks; ++walk) {
      int x = p % nx, y = p / nx;
      for (int step = 0; step < max_steps; ++step) {
        switch (direction(rng)) {
          case 0: ++x; break;
          case 1: --x; break;
          case 2: ++y; break;
          default: --y; break;
        }
        const int q = y * nx + x;
        if (boundary[q]) {
          sum += values[q];
          ++hits;
          break;
        }
      }
    }
    truncated_walks += walks - hits;
    u[p] = hits ? static_cast<float>(sum / hits) : NAN;
  }

  const float dx = 1.0f / (nx - 1), dy = 1.0f / (ny - 1);
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      const int p = y * nx + x;
      const int xl = std::max(0, x - 1), xr = std::min(nx - 1, x + 1);
      const int yd = std::max(0, y - 1), yu = std::min(ny - 1, y + 1);
      ex[p] = -(u[y * nx + xr] - u[y * nx + xl]) / ((xr - xl) * dx);
      ey[p] = -(u[yu * nx + x] - u[yd * nx + x]) / ((yu - yd) * dy);
    }
  }

  std::ofstream out("field.csv");
  out << "x,y,u,u_true,u_error,abs_u_error,Ex,Ey,Ex_true,Ey_true,E_error\n";
  double squared_error_sum = 0.0;
  float max_abs_error = 0.0f;
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      const int p = y * nx + x;
      const float px = x * dx, py = y * dy;
      const float truth = exact_u(px, py);
      const float error = u[p] - truth;
      const float ex_true = -2.0f * px, ey_true = 2.0f * py;
      const float e_error = std::hypot(ex[p] - ex_true, ey[p] - ey_true);
      squared_error_sum += static_cast<double>(error) * error;
      max_abs_error = std::max(max_abs_error, std::abs(error));
      out << px << ',' << py << ',' << u[p] << ',' << truth << ',' << error
          << ',' << std::abs(error) << ',' << ex[p] << ',' << ey[p] << ','
          << ex_true << ',' << ey_true << ',' << e_error << '\n';
    }
  }
  const double rmse = std::sqrt(squared_error_sum / n);
  std::cout << "CPU: wrote field.csv (" << nx << " x " << ny << ", " << walks
            << " walks/point), potential RMSE=" << rmse
            << ", max_abs_error=" << max_abs_error
            << ", truncated_walks=" << truncated_walks << '\n';
}


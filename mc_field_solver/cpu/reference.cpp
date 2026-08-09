#include "monte_carlo.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>

Solution solve_cpu(const Domain& d, int num_walks, std::uint64_t seed) {
    if (num_walks <= 0) throw std::invalid_argument("walks must be positive");
    Solution out;
    const std::size_t count = static_cast<std::size_t>(d.nx) * d.ny;
    out.potential = d.boundary_value;
    out.standard_error.assign(count, 0.0f);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> direction(0, 3);
    const auto begin = std::chrono::steady_clock::now();
    for (int start_idx : d.interior_indices) {
        double sum = 0.0;
        double sum2 = 0.0;
        const int sx = start_idx % d.nx;
        const int sy = start_idx / d.nx;
        for (int walk = 0; walk < num_walks; ++walk) {
            int x = sx, y = sy;
            while (d.cell_type[y * d.nx + x] !=
                   static_cast<std::uint8_t>(CellType::Boundary)) {
                switch (direction(rng)) {
                    case 0: ++x; break;
                    case 1: --x; break;
                    case 2: ++y; break;
                    default: --y; break;
                }
            }
            const double value = d.boundary_value[y * d.nx + x];
            sum += value;
            sum2 += value * value;
        }
        const double mean = sum / num_walks;
        const double variance = std::max(0.0, sum2 / num_walks - mean * mean);
        out.potential[start_idx] = static_cast<float>(mean);
        out.standard_error[start_idx] = static_cast<float>(std::sqrt(variance / num_walks));
    }
    out.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    compute_field(d, out);
    return out;
}

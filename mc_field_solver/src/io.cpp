#include "monte_carlo.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

void compute_field(const Domain& d, Solution& s) {
    const std::size_t count = static_cast<std::size_t>(d.nx) * d.ny;
    s.ex.assign(count, 0.0f);
    s.ey.assign(count, 0.0f);
    s.magnitude.assign(count, 0.0f);
    for (int y = 1; y < d.ny - 1; ++y) {
        for (int x = 1; x < d.nx - 1; ++x) {
            const int i = y * d.nx + x;
            const float ex = -(s.potential[i + 1] - s.potential[i - 1]) / (2.0f * d.dx);
            const float ey = -(s.potential[i + d.nx] - s.potential[i - d.nx]) / (2.0f * d.dy);
            s.ex[i] = ex;
            s.ey[i] = ey;
            s.magnitude[i] = std::sqrt(ex * ex + ey * ey);
        }
    }
}

static void write_csv(const std::filesystem::path& path, const std::vector<float>& data,
                      int nx, int ny) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open output file: " + path.string());
    f << std::setprecision(8);
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            if (x) f << ',';
            f << data[y * nx + x];
        }
        f << '\n';
    }
}

void write_results(const std::string& directory, const Domain& d, const Solution& s) {
    const std::filesystem::path dir(directory);
    std::filesystem::create_directories(dir);
    write_csv(dir / "potential.csv", s.potential, d.nx, d.ny);
    write_csv(dir / "stderr.csv", s.standard_error, d.nx, d.ny);
    write_csv(dir / "Ex.csv", s.ex, d.nx, d.ny);
    write_csv(dir / "Ey.csv", s.ey, d.nx, d.ny);
    write_csv(dir / "E.csv", s.magnitude, d.nx, d.ny);
}

double linear_rmse(const Domain& d, const Solution& s) {
    double squared = 0.0;
    for (int idx : d.interior_indices) {
        const double exact = static_cast<double>(idx % d.nx) / (d.nx - 1);
        const double error = s.potential[idx] - exact;
        squared += error * error;
    }
    return std::sqrt(squared / d.interior_indices.size());
}

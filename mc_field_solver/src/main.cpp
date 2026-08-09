#include "monte_carlo.h"
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

struct Options {
    int nx = 64, ny = 64, walks = 4096, block_size = 256;
    std::uint64_t seed = 123456789ULL;
    std::string backend = "cpu", output = "output";
    double max_rmse = -1.0;
};

static Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            std::cout << "Usage: mc_field_solver [--nx N] [--ny N] [--walks N] "
                         "[--backend cpu|cuda] [--block-size N] [--seed N] "
                         "[--output DIR] [--max-rmse X]\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--nx") o.nx = std::stoi(value);
        else if (key == "--ny") o.ny = std::stoi(value);
        else if (key == "--walks") o.walks = std::stoi(value);
        else if (key == "--block-size") o.block_size = std::stoi(value);
        else if (key == "--seed") o.seed = std::stoull(value);
        else if (key == "--backend") o.backend = value;
        else if (key == "--output") o.output = value;
        else if (key == "--max-rmse") o.max_rmse = std::stod(value);
        else throw std::invalid_argument("unknown option: " + key);
    }
    return o;
}

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        const Domain domain = make_linear_rectangle(o.nx, o.ny);
        Solution solution;
        if (o.backend == "cpu") {
            solution = solve_cpu(domain, o.walks, o.seed);
        } else if (o.backend == "cuda") {
#ifdef MC_HAVE_CUDA
            solution = solve_cuda(domain, o.walks, o.seed, o.block_size);
#else
            throw std::runtime_error("this build has no CUDA backend");
#endif
        } else {
            throw std::invalid_argument("backend must be cpu or cuda");
        }
        const double rmse = linear_rmse(domain, solution);
        write_results(o.output, domain, solution);
        std::cout << "backend=" << o.backend << " grid=" << o.nx << 'x' << o.ny
                  << " walks=" << o.walks << " runtime_ms=" << solution.elapsed_ms
                  << " rmse=" << rmse << " output=" << o.output << '\n';
        if (o.max_rmse >= 0.0 && rmse > o.max_rmse) return 2;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}

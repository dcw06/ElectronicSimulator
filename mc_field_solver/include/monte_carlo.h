#pragma once

#include "domain.h"
#include <cstdint>
#include <string>
#include <vector>

struct Solution {
    std::vector<float> potential;
    std::vector<float> standard_error;
    std::vector<float> ex;
    std::vector<float> ey;
    std::vector<float> magnitude;
    double elapsed_ms = 0.0;
};

Solution solve_cpu(const Domain& domain, int num_walks, std::uint64_t seed);
#ifdef MC_HAVE_CUDA
Solution solve_cuda(const Domain& domain, int num_walks, std::uint64_t seed,
                    int block_size);
#endif
#ifdef MC_HAVE_MUSA
Solution solve_musa(const Domain& domain, int num_walks, std::uint64_t seed,
                    int block_size);
#endif

void compute_field(const Domain& domain, Solution& solution);
void write_results(const std::string& directory, const Domain& domain,
                   const Solution& solution);
double linear_rmse(const Domain& domain, const Solution& solution);

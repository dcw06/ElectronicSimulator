#pragma once

#include <cstdint>
#include <vector>

enum class CellType : std::uint8_t { Outside = 0, Interior = 1, Boundary = 2 };

struct Domain {
    int nx = 0;
    int ny = 0;
    float dx = 1.0f;
    float dy = 1.0f;
    std::vector<std::uint8_t> cell_type;
    std::vector<float> boundary_value;
    std::vector<int> interior_indices;
};

Domain make_linear_rectangle(int nx, int ny);

#include "domain.h"
#include <stdexcept>

Domain make_linear_rectangle(int nx, int ny) {
    if (nx < 3 || ny < 3) throw std::invalid_argument("nx and ny must be >= 3");
    Domain d;
    d.nx = nx;
    d.ny = ny;
    d.dx = 1.0f / static_cast<float>(nx - 1);
    d.dy = 1.0f / static_cast<float>(ny - 1);
    d.cell_type.resize(static_cast<std::size_t>(nx) * ny);
    d.boundary_value.assign(static_cast<std::size_t>(nx) * ny, 0.0f);
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const int idx = y * nx + x;
            const bool boundary = x == 0 || x == nx - 1 || y == 0 || y == ny - 1;
            d.cell_type[idx] = static_cast<std::uint8_t>(
                boundary ? CellType::Boundary : CellType::Interior);
            if (boundary) {
                // u(0,y)=0, u(1,y)=1, u(x,0)=u(x,1)=x -> exact u(x,y)=x.
                d.boundary_value[idx] = static_cast<float>(x) * d.dx;
            } else {
                d.interior_indices.push_back(idx);
            }
        }
    }
    return d;
}

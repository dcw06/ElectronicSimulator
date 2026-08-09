cmake_minimum_required(VERSION 3.24)
project(mc_field_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(MC_ENABLE_CUDA "Build the CUDA backend" ON)

add_executable(mc_field_solver
    src/main.cpp
    src/domain.cpp
    src/io.cpp
    cpu/reference.cpp
)
target_include_directories(mc_field_solver PRIVATE include)

if(MC_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        target_sources(mc_field_solver PRIVATE kernels/monte_carlo.cu)
        target_link_libraries(mc_field_solver PRIVATE CUDA::cudart CUDA::curand)
        target_compile_definitions(mc_field_solver PRIVATE MC_HAVE_CUDA=1)
        set_target_properties(mc_field_solver PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
        )
    else()
        message(WARNING "CUDA compiler unavailable; building CPU backend only")
    endif()
endif()

enable_testing()
add_test(NAME linear_boundary_cpu
    COMMAND mc_field_solver --nx 24 --ny 24 --walks 2000 --backend cpu
            --output ${CMAKE_CURRENT_BINARY_DIR}/test_output --max-rmse 0.06)

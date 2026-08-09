cmake_minimum_required(VERSION 3.24)
project(cuda_walk_on_grid LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(mc_laplace mc_laplace.cu)
target_compile_options(mc_laplace PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)


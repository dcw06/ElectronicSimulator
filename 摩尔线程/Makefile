CUDA_NVCC ?= nvcc
MUSA_HOME ?= /usr/local/musa
MUSA_MCC ?= $(MUSA_HOME)/bin/mcc
CXXFLAGS ?= -O3 -std=c++17

.PHONY: cpu cuda musa plot clean

cpu: mc_laplace_cpu

mc_laplace_cpu: mc_laplace_cpu.cpp
	$(CXX) $(CXXFLAGS) mc_laplace_cpu.cpp -o $@

cuda: mc_laplace_cuda

musa: mc_laplace_musa

plot:
	@test -f field.csv || { echo "field.csv not found; run the solver first"; exit 1; }
	@if command -v Rscript >/dev/null 2>&1; then \
		Rscript visualize.R field.csv; \
	else \
		python3 visualize.py field.csv; \
	fi

mc_laplace_cuda: mc_laplace.cu gpu_backend.h
	$(CUDA_NVCC) $(CXXFLAGS) --use_fast_math mc_laplace.cu -o $@

mc_laplace_musa: mc_laplace.cu gpu_backend.h
	$(MUSA_MCC) -x musa $(CXXFLAGS) -DUSE_MUSA \
		-I$(MUSA_HOME)/include -L$(MUSA_HOME)/lib \
		mc_laplace.cu -lmurand -lmusart -o $@

clean:
	$(RM) mc_laplace_cpu mc_laplace_cuda mc_laplace_musa field.csv field_validation.png

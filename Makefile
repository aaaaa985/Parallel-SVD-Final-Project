CXX := g++
NVCC := nvcc

OMP_SCHEDULE_KIND ?= 3

CXXFLAGS := -O3 -std=c++17 -DUSE_OPENMP -DOMP_SCHEDULE_KIND=$(OMP_SCHEDULE_KIND) -fopenmp -pthread
NVCCFLAGS := -O3 -std=c++17 -arch=sm_86

CPU_TARGET := svd_cpu
GPU_TARGET := svd_gpu

.PHONY: all cpu gpu clean

all: cpu gpu

cpu:
	$(CXX) $(CXXFLAGS) main.cpp bidiagonalization.cpp gkh.cpp -o $(CPU_TARGET)

gpu: main_gpu.o bidiagonalization_cpu.o gkh_cpu.o bidiagonalization_cuda.o
	$(NVCC) main_gpu.o bidiagonalization_cpu.o gkh_cpu.o bidiagonalization_cuda.o -lcublas -lgomp -lpthread -o $(GPU_TARGET)

main_gpu.o: main.cpp bidiagonalization_cuda.h
	$(CXX) $(CXXFLAGS) -DUSE_CUDA_BIDIAG -c main.cpp -o main_gpu.o

bidiagonalization_cpu.o: bidiagonalization.cpp
	$(CXX) $(CXXFLAGS) -c bidiagonalization.cpp -o bidiagonalization_cpu.o

gkh_cpu.o: gkh.cpp
	$(CXX) $(CXXFLAGS) -c gkh.cpp -o gkh_cpu.o

bidiagonalization_cuda.o: bidiagonalization_cuda.cu bidiagonalization_cuda.h
	$(NVCC) $(NVCCFLAGS) -c bidiagonalization_cuda.cu -o bidiagonalization_cuda.o

clean:
	rm -f *.o $(CPU_TARGET) $(GPU_TARGET)
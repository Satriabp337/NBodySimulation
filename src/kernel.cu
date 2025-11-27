#include <cuda_runtime.h>
#include <stdio.h>

void printCudaInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("TIDAK ADA GPU CUDA TERDETEKSI!\n");
    } else {
        printf("Sukses! Terdeteksi %d GPU CUDA.\n", deviceCount);
    }
}
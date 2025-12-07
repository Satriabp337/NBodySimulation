#include "physics.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

//alokasi memory gppu
extern "C" void allocatedDeviceMemory (Particle** d_particles, size_t size){
    
    cudaError_t err = cudaMalloc((void**)d_particles, size);
    
    if(err != cudaSuccess){
        printf("error cudaMalloc: %s\n", cudaGetErrorString(err));
    }
}

//transfer memory
extern "C" void copyToDevice (Particle* d_particles, Particle* h_particles, size_t size){
    cudaMemcpy(d_particles, h_particles, size, cudaMemcpyHostToDevice);
}

extern "C" void copyFromDevice (Particle* h_particles, Particle* d_particles, size_t size ){
    cudaMemcpy(h_particles, d_particles, size, cudaMemcpyDeviceToHost);
}

//clear memory
extern "C" void freeDeviceMemory (Particle* d_particles){
    cudaFree(d_particles);
}


__global__ void bodyForce (Particle* p, float dt, int n, float mouseX, float mouseY, bool isPressed){
    
    //alokasi shared memory 
    __shared__ float3 sharedData [BLOCK_SIZE];

    int tid =  threadIdx.x;     //id local (0-255)
    int i = blockIdx.x * blockDim.x + threadIdx.x; //id global

    //simpan data diregister
    float2 myPos;
    float2 myVel;

    //load data vram
    if(i < n) {
        myPos = p[i].pos;
        myVel = p[i].vel;
    }

    float accX = 0.0f;
    float accY = 0.0f;

    //logika mouse
    if (i < n && isPressed) {
        float dx = mouseX - myPos.x;
        float dy = mouseY - myPos.y;
        float distSq = dx*dx + dy*dy + 500.0f;
        float f = 50000.0f / (distSq * sqrtf(distSq));
        accX += f * dx;
        accY += f * dy;
    }   

    //loop tiling
    for(int tile = 0; tile < gridDim.x; tile++) {
        
        //index global
        int idx = tile * blockDim.x + tid;

        if (idx < n) {
            //ambil data dari vram
            sharedData[tid] = make_float3(p[idx].pos.x, p[idx].pos.y, p[idx].mass);
        } else {
            sharedData[tid] = make_float3(0.0f, 0.0f, 0.0f);
        }

        __syncthreads();

        if (i < n) {
            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE; k++) {
                float3 other = sharedData[k];

                float dx = other.x - myPos.x;
                float dy = other.y - myPos.y;
                float distSq = dx*dx + dy*dy + SOFTENING*SOFTENING;

                float f = (G_CONST * other.z) / (distSq * sqrtf(distSq));

                accX += f * dx;
                accY += f * dy;
            }
        }

        __syncthreads();
    }

    //update posisi akhir
    if (i < n) {
        myVel.x += accX * dt;
        myVel.y += accY * dt;

        p[i].vel = myVel;
        p[i].pos.x += myVel.x * dt;
        p[i].pos.y += myVel.y * dt;
    }
}

extern "C" void launchCudaBody (Particle* d_particles, int n, int blocks, int threads, float mouseX, float mouseY, bool isPressed) {

    //konfigurasi grid
    bodyForce <<<blocks, threads>>> (d_particles, DT, n, mouseX, mouseY, isPressed);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Kernel Error : %s\n", cudaGetErrorString(err));
    }
}
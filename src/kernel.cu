#include "physics.h"
#include <cuda_runtime.h>
#include <stdio.h>

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

__global__ void bodyForce (Particle* p, float dt, int n){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n){
        float dx, dy, distSq, dist, f;
        float total_fx = 0.0f;
        float total_fy = 0.0f;

        //looping interaksi dengan partikel lain
        for(int j = 0; j < n; j++){
            if (i == j) continue;

            //hitung vektor
            dx = p[j].pos.x - p[i].pos.x;
            dy = p[j].pos.y - p[i].pos.y;

            //hitung jarak
            distSq = dx*dx + dy*dy + SOFTENING*SOFTENING;
            dist = sqrt(distSq);

            //hitung rumus newton
            f = (G_CONST * p[j].mass) / (dist * distSq);

            total_fx += f * dx;
            total_fy += f * dy;
        }

        //update
        p[i].vel.x += total_fx * dt;
        p[i].vel.y += total_fy * dt;

        p[i].pos.x += p[i].vel.x * dt;
        p[i].pos.y += p[i].vel.y * dt;
    }
}

extern "C" void launchCudaBody (Particle* d_particles, int n, int blocks, int threads) {

    //konfigurasi grid
    bodyForce <<<blocks, threads>>> (d_particles, DT, n);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Kernel Error : %s\n", cudaGetErrorString(err));
    }
}
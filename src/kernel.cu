#include "physics.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

extern "C" void allocatedDeviceMemory (Particle** d_particles, size_t size){
    cudaError_t err = cudaMalloc((void**)d_particles, size);
    if(err != cudaSuccess){
        printf("error cudaMalloc: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void copyToDevice (Particle* d_particles, Particle* h_particles, size_t size){
    cudaMemcpy(d_particles, h_particles, size, cudaMemcpyHostToDevice);
}

extern "C" void copyFromDevice (Particle* h_particles, Particle* d_particles, size_t size ){
    cudaMemcpy(h_particles, d_particles, size, cudaMemcpyDeviceToHost);
}

extern "C" void freeDeviceMemory (Particle* d_particles){
    cudaFree(d_particles);
}

__global__ void bodyForce (Particle* p, float dt, int n, float mouseX, float mouseY, bool isPressed){
    __shared__ float3 sharedData [BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float2 myPos, myVel;
    if(i < n){ myPos = p[i].pos; myVel = p[i].vel; }

    float accX = 0, accY = 0;

    if (i<n && isPressed){
        float dx = mouseX - myPos.x;
        float dy = mouseY - myPos.y;
        float distSq = dx*dx + dy*dy + 500;
        float f = 50000.0f / (distSq * sqrtf(distSq));
        accX += f*dx; accY += f*dy;
    }

    for(int tile=0; tile<gridDim.x; tile++){
        int idx = tile*blockDim.x + tid;

        sharedData[tid] = (idx < n) ? make_float3(p[idx].pos.x,p[idx].pos.y,p[idx].mass)
                                    : make_float3(0,0,0);

        __syncthreads();

        if(i < n){
            #pragma unroll
            for(int k=0; k < BLOCK_SIZE; k++){
                float3 other = sharedData[k];
                float dx = other.x - myPos.x;
                float dy = other.y - myPos.y;
                float distSq = dx*dx + dy*dy + SOFTENING*SOFTENING;
                float force = (G_CONST * other.z) / (distSq * sqrtf(distSq));

                accX += force*dx;
                accY += force*dy;
            }
        }
        __syncthreads();
    }

    if (i<n){
        myVel.x += accX * dt;
        myVel.y += accY * dt;
        p[i].vel = myVel;
        p[i].pos.x += myVel.x * dt;
        p[i].pos.y += myVel.y * dt;
    }
}

extern "C" void launchCudaBody(Particle* d_particles,int n,int blocks,int threads,float mouseX,float mouseY,bool isPressed){
    bodyForce<<<blocks,threads>>>(d_particles,DT,n,mouseX,mouseY,isPressed);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if(err!=cudaSuccess) printf("Kernel Error : %s\n", cudaGetErrorString(err));
}

#ifndef PHYSICS_H
#define PHYSICS_H

#include <cuda_runtime.h>

#define G_CONST 1.0f
#define SOFTENING 5.0f
#define DT 0.01f

struct Particle {
    float2 pos;
    float2 vel;
    float mass;
    int colorVal;
};

void cpuBodyInteraction(Particle* particles, int n);

extern "C" void launchCudaBody(Particle* d_particles, int n, int blocks, int threads, float mouseX, float mouseY, bool isPressed);
extern "C" void allocatedDeviceMemory (Particle** d_particles, size_t size);
extern "C" void copyToDevice (Particle* d_particles, Particle* h_particles, size_t size);
extern "C" void copyFromDevice (Particle* h_particles, Particle* d_particles, size_t size);
extern "C" void freeDeviceMemory (Particle* d_particles);

#endif

#ifndef PHYSICS_H
#define PHYSICS_H

#ifdef __CUDACC__
    #define ENABLE_CUDA
#endif

#ifdef ENABLE_CUDA
    #include <cuda_runtime.h> // Cuma diload kalau ada CUDA
#else
    struct float2 { float x, y; }; 
#endif

#define G_CONST 1.0f //konstanta gravitasi
#define SOFTENING 5.0f
#define DT 0.1f //waktu

struct Particle {
    float2 pos; //x, y position
    float2 vel; //x, y velocity
    float mass;

    int colorVal;
};

//Fungction CPU
void cpuBodyInteraction(Particle* particles, int n);
//Fungction OPENMP
void cpuBodyInteractionOpenMP(Particle* particles, int n);

#ifdef ENABLE_CUDA
//Fungction Wrapper GPU
extern "C" void launchCudaBody(Particle* d_particles, int n, int blocks, int threads, float mouseX, float mouseY, bool isPressed);
//inisiasi memory GPU
extern "C" void allocatedDeviceMemory (Particle** d_particles, size_t size);
extern "C" void copyToDevice (Particle* d_particles, Particle* h_particles, size_t size);
extern "C" void copyFromDevice (Particle* h_particles, Particle* d_particles, size_t size);
extern "C" void freeDeviceMemory (Particle* d_particles);
#endif

#endif
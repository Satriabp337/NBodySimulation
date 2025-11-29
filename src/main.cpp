#include <SFML/Graphics.hpp>    //grafis
#include <vector>               //wadah data dinamis
#include <cmath>                //sin, cos
#include <cstdlib>              //random
#include "physics.h"            

//konfigurasi
const int WINDOW_WIDTH = 1200;
const int WINDOW_HEIGHT = 800;
const int NUM_PARTICLES = 2048;

//fungsi random
float randomFloat () {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void initParticles (std::vector<Particle> & particles) {
    //pusat layar
    float centerX = WINDOW_WIDTH / 2.0f;
    float centerY = WINDOW_HEIGHT / 2.0f;

    for (int i = 0; i < NUM_PARTICLES; i++){
        Particle p;

        //sebaran partikel
        float dist = randomFloat() * 300.0 + 10.0f;

        //lengkungan
        float angle = dist * 0.05f; 
        angle += randomFloat() * 6.28f; //0 - 2phi

        //konversi polar ke cartesian
        p.pos.x = centerX * std::cos(angle) * dist; // x = r * cos (sudut)
        p.pos.y = centerY * std::sin(angle) * dist; // y = r * cos (sudut)

        //kecepatan orbit
        float orbitalVel = std::sqrt(dist) * 0.5f;

        //vektor tegak lurus dari sudut posisi
        p.vel.x = -std::sin(angle) * orbitalVel;
        p.vel.y = std::cos(angle) * orbitalVel;

        //massa
        p.mass = randomFloat() * 4.0f + 1.0f;

        //Masukkan ke vector
        particles.push_back(p);
    }
}

int main () {
    
    //siapkan data di cpu
    std::vector<Particle> host_particles;
    
    //isi data
    initParticles(host_particles);

    //total byte
    size_t size = NUM_PARTICLES * sizeof(Particle);

    //set up gpu
    Particle* d_particles = nullptr;
    allocatedDeviceMemory(&d_particles, size);
    copyToDevice(d_particles, host_particles.data(), size);

    //konfigurasi thread dan block
    int threadsPerBlock = 256;
    int blockPerGrid = (NUM_PARTICLES + threadsPerBlock - 1) / threadsPerBlock;
    
}
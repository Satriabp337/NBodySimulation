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
        p.pos.x = centerX + std::cos(angle) * dist; // x = r * cos (sudut)
        p.pos.y = centerY + std::sin(angle) * dist; // y = r * cos (sudut)

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

    //setup visualisasi
    sf::RenderWindow window (sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "CUDA N-BODY SIMULATION");
    window.setFramerateLimit(60); //batasan 60 fps
    
    sf::VertexArray visualParticles(sf::Points, NUM_PARTICLES);

    while(window.isOpen()){
        sf::Event event;

        while(window.pollEvent(event)){
            if (event.type == sf::Event::Closed)
            window.close();
        }

        //hitung fisika (GPU)
        launchCudaBody(d_particles, NUM_PARTICLES, blockPerGrid, threadsPerBlock);
        
        //copy data gpu ke cpu
        copyFromDevice(host_particles.data(), d_particles, size);

        //update visual
        for(int i = 0; i < NUM_PARTICLES; i++){
            float x = host_particles[i].pos.x;
            float y = host_particles[i].pos.y;
            
            //set posisi
            visualParticles[i].position = sf::Vector2f(x, y);

            //pewarnaan
            float speed = sqrt(host_particles[i].vel.x * host_particles[i].vel.x + 
                                host_particles[i].vel.y * host_particles[i].vel.y);

            int colorVal = std::min(255, (int)(speed * 200.0f)); //mapping kecepatan ke warna
            visualParticles[i].color = sf::Color(255, 255 - colorVal, 255 - colorVal);
        }

        //render
        window.clear(sf::Color::Black); //Hapus layar jadi hitam
        window.draw(visualParticles);   //Gambar partikel
        window.display();               //tampilkan
    }

    //cleanup
    freeDeviceMemory(d_particles);

    return 0;
}
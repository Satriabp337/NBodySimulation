#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "physics.h"

// konfigurasi
const int WINDOW_WIDTH = 1200;
const int WINDOW_HEIGHT = 800;
const int NUM_PARTICLES = 20000;

// fungsi random
float randomFloat() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void initParticles(std::vector<Particle> &particles)
{
    float centerX = 0.0f;
    float centerY = 0.0f;
    float coreMass = 10000.0f;
    float maxDist = 300.0f;

    // Black Hole
    Particle blackHole;
    blackHole.pos.x = centerX;
    blackHole.pos.y = centerY;
    blackHole.vel.x = 0.0f;
    blackHole.vel.y = 0.0f;
    blackHole.mass = coreMass;
    blackHole.colorVal = 255;
    particles.push_back(blackHole);

    // Bintang
    for (int i = 1; i < NUM_PARTICLES; i++)
    {
        Particle p;

        float r = maxDist * std::sqrt(randomFloat());
        if (r < 15.0f) r = 15.0f;

        float theta = randomFloat() * 6.28318f;
        p.pos.x = centerX + r * std::cos(theta);
        p.pos.y = centerY + r * std::sin(theta);

        float v = std::sqrt(G_CONST * coreMass / r);
        v *= (0.8f + (randomFloat() * 0.4f));

        p.vel.x = -std::sin(theta) * v;
        p.vel.y = std::cos(theta) * v;

        p.mass = randomFloat() * 2.0f + 0.5f;
        p.colorVal = 200;

        particles.push_back(p);
    }
}

int main()
{
    std::vector<Particle> host_particles;
    initParticles(host_particles);

    size_t size = NUM_PARTICLES * sizeof(Particle);

    bool useGPU = true;

    // ============= CPU FALLBACK OTOMATIS =============
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        useGPU = false;
        std::cout << "\n[WARNING] CUDA GPU tidak ditemukan!\n";
        std::cout << "Mode otomatis beralih ke CPU.\n\n";
    }
    // ==================================================

    Particle *d_particles = nullptr;

    if (useGPU) {
        allocatedDeviceMemory(&d_particles, size);
        copyToDevice(d_particles, host_particles.data(), size);
    }

    int threadsPerBlock = 256;
    int blockPerGrid = (NUM_PARTICLES + threadsPerBlock - 1) / threadsPerBlock;

    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), 
        useGPU ? "CUDA N-Body Simulation [GPU]" : "N-Body Simulation [CPU]");
    window.setFramerateLimit(60);

    sf::VertexArray visualParticles(sf::Points, NUM_PARTICLES);
    sf::View camera(sf::FloatRect(0, 0, 1200, 800));
    camera.setCenter(0, 0);

    bool isDragging = false;
    sf::Vector2i lastMousePos;

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed) window.close();

            if (event.type == sf::Event::MouseWheelScrolled)
                camera.zoom(event.mouseWheelScroll.delta > 0 ? 0.9f : 1.1f);

            if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Right) {
                isDragging = true; lastMousePos = sf::Mouse::getPosition(window);
            }
            if (event.type == sf::Event::MouseButtonReleased && event.mouseButton.button == sf::Mouse::Right)
                isDragging = false;

            if (event.type == sf::Event::MouseMoved && isDragging)
            {
                sf::Vector2i current = sf::Mouse::getPosition(window);
                sf::Vector2f oldPos = window.mapPixelToCoords(lastMousePos, camera);
                sf::Vector2f newPos = window.mapPixelToCoords(current, camera);
                camera.move(oldPos - newPos);
                lastMousePos = current;
            }

            // Toggle mode (GPU/CPU)
            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Space && deviceCount > 0) {
                useGPU = !useGPU;
                window.setTitle(useGPU ? "CUDA N-Body Simulation [GPU]" : "N-Body Simulation [CPU]");
            }
        }

        sf::Vector2f worldPos = window.mapPixelToCoords(sf::Mouse::getPosition(window));
        bool isPressed = sf::Mouse::isButtonPressed(sf::Mouse::Left);

        if (useGPU) {
            launchCudaBody(d_particles, NUM_PARTICLES, blockPerGrid, threadsPerBlock, worldPos.x, worldPos.y, isPressed);
            copyFromDevice(host_particles.data(), d_particles, size);
        } else {
            cpuBodyInteraction(host_particles.data(), NUM_PARTICLES);
        }

        for (int i = 0; i < NUM_PARTICLES; i++)
        {
            visualParticles[i].position = sf::Vector2f(host_particles[i].pos.x, host_particles[i].pos.y);

            float speed = std::sqrt(host_particles[i].vel.x*host_particles[i].vel.x + host_particles[i].vel.y*host_particles[i].vel.y);
            sf::Uint8 r, g, b, a;

            if (speed < 5.0f) { r = 80 + speed*20; g = 0; b = 200; a = 30; }
            else if (speed < 12.0f) { r = 0; g = 150 + speed*5; b = 255; a = 60; }
            else { r = g = b = 255; a = 150; }

            visualParticles[i].color = sf::Color(r,g,b,a);
        }

        window.setView(camera);
        window.clear(sf::Color::Black);
        window.draw(visualParticles, sf::RenderStates(sf::BlendAdd));
        window.display();
    }

    if (useGPU) freeDeviceMemory(d_particles);
    return 0;
}

#include <SFML/Graphics.hpp> //grafis
#include <iomanip> 
#include <sstream> 
#include <vector>            //wadah data dinamis
#include <cmath>             //sin, cos
#include <cstdlib>           //random
#include <iostream>
#include "physics.h"

// konfigurasi
const int WINDOW_WIDTH = 1600;
const int WINDOW_HEIGHT = 1200;
const int NUM_PARTICLES = 10000;

enum SimulationMode {CPU_SERIAL, CPU_OPENMP, GPU_CUDA};
 

// fungsi random
float randomFloat()
{
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void createGalaxy(std::vector<Particle> &particles, int count, float offsetX, float offsetY, float velX, float velY, float coreMass, float radius, int colorBase) {
    Particle blackhole;
    blackhole.pos.x = offsetX;
    blackhole.pos.y = offsetY;
    blackhole.vel.x = velX;
    blackhole.vel.y = velY;
    blackhole.mass = coreMass;
    blackhole.colorVal = 255;
    particles.push_back(blackhole);

    for (int i = 1; i < count; i++) {
        Particle p;

        // sebar partikel
        float r = radius * std::sqrt(randomFloat());
        if (r < 10.0f) r = 10.0f;

        float theta = randomFloat() * 6.28318f;

        //posisi relatif terhadap pusat galaxy
        float localX = r * std::cos(theta);  
        float localY = r * std::sin(theta);

        //posisi global
        p.pos.x = offsetX + localX;
        p.pos.y = offsetY + localY;

        //kecepatan orbit
        float v = std::sqrt(G_CONST * coreMass / r);
        v *=(0.8f + randomFloat() * 0.4f);

        //kecepatan total
        p.vel.x = velX - std::sin(theta) * v;
        p.vel.y = velY + std::cos(theta) * v;

        p.mass = randomFloat() * 2.0f + 0.5f;
        p.colorVal = colorBase;

        particles.push_back(p);
 
    }

}

void initParticles(std::vector<Particle> &particles) {
    particles.clear();

    int halfParticles = NUM_PARTICLES/2;

    createGalaxy(particles, halfParticles, 
                 -350.0f, 0.0f,         //posisi
                 1.0f, 0.4f,            //kecepatan gerak
                 8000.0f, 350.0f,      //massa dan radius
                 100);                  //warna

    createGalaxy(particles, NUM_PARTICLES - halfParticles, 
                 350.0f, 0.0f,         //posisi
                 -1.0f, -0.4f,            //kecepatan gerak
                 8000.0f, 350.0f,      //massa dan radius
                 200);                  //warna
}

int main()
{

    // siapkan data di cpu
    std::vector<Particle> host_particles;

    // isi data
    initParticles(host_particles);

    // total byte
    size_t size = NUM_PARTICLES * sizeof(Particle);

    // set up gpu
    Particle *d_particles = nullptr;
    allocatedDeviceMemory(&d_particles, size);
    copyToDevice(d_particles, host_particles.data(), size);

    // konfigurasi thread dan block
    int threadsPerBlock = 256;
    int blockPerGrid = (NUM_PARTICLES + threadsPerBlock - 1) / threadsPerBlock;

    // setup visualisasi
    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "CUDA N-BODY SIMULATION");
    window.setFramerateLimit(60); // batasan 60 fps

    sf::VertexArray visualParticles(sf::Points, NUM_PARTICLES);

    sf::View camera(sf::FloatRect(0, 0, 1200, 800));
    camera.setCenter(0, 0);

    bool isDragging = false;
    sf::Vector2i lastMousePos;

    SimulationMode currentMode = GPU_CUDA;

    //HUD
    sf::Font font;
    if(!font.loadFromFile("arial.ttf")) {
        std::cout << "gagal load font" << std::endl;
    }

    sf::Text statsText;
    statsText.setFont(font);
    statsText.setCharacterSize(18);
    statsText.setFillColor(sf::Color::White);
    statsText.setPosition(10.0f, 10.0f);

    sf::Clock fpsClock;         //hitung fps render
    sf::Clock physicsClock;    //hitung durasi


    while (window.isOpen())
    {
        sf::Event event;

        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
            {
                window.close();
            }

            // zoom
            if (event.type == sf::Event::MouseWheelScrolled)
            {
                if (event.mouseWheelScroll.delta > 0)
                {
                    camera.zoom(0.9f);
                }
                else
                {
                    camera.zoom(1.1f);
                }
            }

            // drag
            if (event.type == sf::Event::MouseButtonPressed)
            {
                if (event.mouseButton.button == sf::Mouse::Right)
                {
                    isDragging = true;
                    lastMousePos = sf::Mouse::getPosition(window); // simpan posisi
                }
            }
            else if (event.type == sf::Event::MouseButtonReleased)
            {
                if (event.mouseButton.button == sf::Mouse::Right)
                {
                    isDragging = false;
                }
            }
            else if (event.type == sf::Event::MouseMoved)
            {
                if (isDragging)
                {
                    sf::Vector2i currentMousePos = sf::Mouse::getPosition(window);

                    sf::Vector2f oldPos = window.mapPixelToCoords(lastMousePos, camera);
                    sf::Vector2f newPos = window.mapPixelToCoords(currentMousePos, camera);
                    sf::Vector2f delta = oldPos - newPos;

                    camera.move(delta);
                    lastMousePos = currentMousePos;
                }
            }

            // toggle
            if (event.type == sf::Event::KeyPressed)
            {
                if (event.key.code == sf::Keyboard::Num1) {
                    currentMode = GPU_CUDA;
                    window.setTitle("Mode : GPU CUDA");
                } else if (event.key.code == sf::Keyboard::Num2) {
                    currentMode = CPU_SERIAL;
                    window.setTitle("Mode : CPU SERIAL");
                } else if (event.key.code == sf::Keyboard::Num3) {
                    currentMode = CPU_OPENMP;
                    window.setTitle("Mode : CPU OPENMP");
                }
            }
        }

        sf::Vector2i pixelPos = sf::Mouse::getPosition(window);

        sf::Vector2f worldPos = window.mapPixelToCoords(pixelPos, camera);

        bool isPressed = sf::Mouse::isButtonPressed(sf::Mouse::Left);

        //hitung waktu fisika
        physicsClock.restart();

        if (currentMode == GPU_CUDA)
        {
            // hitung fisika (GPU)
            launchCudaBody(d_particles, NUM_PARTICLES, blockPerGrid, threadsPerBlock, worldPos.x, worldPos.y, isPressed);

            // copy data gpu ke cpu
            copyFromDevice(host_particles.data(), d_particles, size);
        }
        else if (currentMode == CPU_SERIAL)
        {
            cpuBodyInteraction(host_particles.data(), NUM_PARTICLES);
        } else if (currentMode == CPU_OPENMP) {
            cpuBodyInteractionOpenMP(host_particles.data(), NUM_PARTICLES);
        }

        //stop stopwatch
        float physicsTime = physicsClock.getElapsedTime().asMilliseconds();
        
        //hitung fps
        float fps = 1.0f / fpsClock.restart().asSeconds();

        //update HUD
        std::string mode;
        if(currentMode == GPU_CUDA) mode = "GPU CUDA";
        else if (currentMode == CPU_SERIAL) mode = "CPU SERIAL";
        else mode = "CPU OPENMP";

        std::stringstream ss;
        ss << "Mode: " << mode << "\n"
           << "Particles: " << NUM_PARTICLES << "\n"
           << "FPS: " << std::fixed << std::setprecision(1) << fps <<"\n"   
           << "Physics Time: " << std::setprecision(2) << physicsTime <<"ms";

        statsText.setString(ss.str());

        // update visual
        for (int i = 0; i < NUM_PARTICLES; i++)
        {   
            float x = host_particles[i].pos.x;
            float y = host_particles[i].pos.y;
            float vx = host_particles[i].vel.x;
            float vy = host_particles[i].vel.y;
            float speed = std::sqrt(vx * vx + vy * vy);
            int baseColor = host_particles[i].colorVal;

            if (baseColor == 255) {
                visualParticles[i].color = sf::Color::White;
            }

            else if (baseColor == 100) {
                visualParticles[i].color = sf::Color(speed*10, speed*20, 255, 100);
            } 
            
            else {
                visualParticles[i].color = sf::Color(255, speed*20, 50, 100);
            }

            visualParticles[i].position = sf::Vector2f(x, y);
        }

        // render
        window.clear(sf::Color::Black); //bakcground

        window.setView(camera);

        sf::RenderStates states;
        states.blendMode = sf::BlendAdd;
        window.draw(visualParticles, states); // Gambar partikel

        window.setView(window.getDefaultView());
        window.draw(statsText);               // Gambar HUD
        window.display();                     // tampilkan
    }

    // cleanup
    freeDeviceMemory(d_particles);

    return 0;
}
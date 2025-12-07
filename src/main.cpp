#include <SFML/Graphics.hpp> //grafis
#include <vector>            //wadah data dinamis
#include <cmath>             //sin, cos
#include <cstdlib>           //random
#include <algorithm>
#include "physics.h"

// konfigurasi
const int WINDOW_WIDTH = 1200;
const int WINDOW_HEIGHT = 800;
const int NUM_PARTICLES = 10000;

enum SimulationMode {CPU_SERIAL, CPU_OPENMP, GPU_CUDA};
 

// fungsi random
float randomFloat()
{
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void initParticles(std::vector<Particle> &particles)
{
    float centerX = 0.0f;
    float centerY = 0.0f;
    float coreMass = 10000.0f; // Massa Black Hole
    float maxDist = 300.0f;    // Jari-jari piringan

    // 1. Black Hole
    Particle blackHole;
    blackHole.pos.x = centerX;
    blackHole.pos.y = centerY;
    blackHole.vel.x = 0.0f;
    blackHole.vel.y = 0.0f;
    blackHole.mass = coreMass;
    blackHole.colorVal = 255; // Putih Terang
    particles.push_back(blackHole);

    // 2. Bintang-bintang
    for (int i = 1; i < NUM_PARTICLES; i++)
    {
        Particle p;

        // --- LOGIKA DISK (PIRINGAN) ---
        float r = maxDist * std::sqrt(randomFloat());
        if (r < 15.0f)
            r = 15.0f; // Jarak minimal

        float theta = randomFloat() * 6.28318f; // Sudut Acak

        // Posisi
        p.pos.x = centerX + r * std::cos(theta);
        p.pos.y = centerY + r * std::sin(theta);

        // --- KECEPATAN ORBIT (PERBAIKAN DI SINI) ---
        // Gunakan G_CONST (1.0f) agar seimbang dengan tarikan gravitasi simulasi
        float v = std::sqrt(G_CONST * coreMass / r);

        // Variasi orbit (biar agak lonjong/alami)
        float variation = 0.8f + (randomFloat() * 0.4f);
        v *= variation;

        // Arah Velocity (Tegak Lurus)
        p.vel.x = -std::sin(theta) * v;
        p.vel.y = std::cos(theta) * v;

        p.mass = randomFloat() * 2.0f + 0.5f;
        p.colorVal = 200; // Init warna default

        particles.push_back(p);
    }
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

        sf::Vector2f worldPos = window.mapPixelToCoords(pixelPos);

        bool isPressed = sf::Mouse::isButtonPressed(sf::Mouse::Left);

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

        // update visual
        for (int i = 0; i < NUM_PARTICLES; i++)
        {
            // 1. Ambil posisi & kecepatan dari CPU (host_particles)
            float x = host_particles[i].pos.x;
            float y = host_particles[i].pos.y;
            float vx = host_particles[i].vel.x;
            float vy = host_particles[i].vel.y;

            // 2. Set Posisi Visual
            visualParticles[i].position = sf::Vector2f(x, y);

            // 3. Hitung Kecepatan (Speed)
            float speed = std::sqrt(vx * vx + vy * vy);

            // 4. Logika Warna "Zarro Style" (Ungu -> Cyan -> Putih)
            sf::Uint8 r, g, b, a;

            // Tuning angka "5.0f" dan "15.0f" ini sesuai kecepatan rata-rata partikelmu.
            // Jika partikelmu lambat, kecilkan angkanya (misal 2.0f dan 8.0f).

            if (speed < 5.0f)
            {
                // PINGGIRAN: MERAH BATA / ORANYE GELAP
                r = 150 + (int)(speed * 20); // 150 -> 250
                g = 50 + (int)(speed * 10);  // 50 -> 100
                b = 10;                      // Sedikit biru

                a = (NUM_PARTICLES < 2000) ? 200 : 80;
            }
            else if (speed < 12.0f)
            {
                // TENGAH: EMAS / KUNING CERAH
                r = 255;
                g = 150 + (int)(speed * 8); // 150 -> 240 (Makin kuning)
                b = 50;

                a = (NUM_PARTICLES < 2000) ? 200 : 100;
            }
            else
            {
                // INTI: PUTIH KEKUNINGAN (SILAU)
                r = 255;
                g = 255;
                b = 200; // Sedikit kuning
                a = 255; // Solid
            }

            visualParticles[i].color = sf::Color(r, g, b, a);
        }

        // render
        window.setView(camera);
        window.clear(sf::Color::Black); // Hapus layar jadi hitam
        sf::RenderStates states;
        states.blendMode = sf::BlendAdd;
        window.draw(visualParticles, states); // Gambar partikel
        window.display();                     // tampilkan
    }

    // cleanup
    freeDeviceMemory(d_particles);

    return 0;
}
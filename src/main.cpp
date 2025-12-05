#include <SFML/Graphics.hpp> //grafis
#include <vector>            //wadah data dinamis
#include <cmath>             //sin, cos
#include <cstdlib>           //random
#include "physics.h"

// konfigurasi
const int WINDOW_WIDTH = 1200;
const int WINDOW_HEIGHT = 800;
const int NUM_PARTICLES = 50000;

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
    float maxDist = 450.0f;    // Jari-jari piringan

    // 1. Black Hole (Wajib ada biar stabil)
    Particle blackHole;
    blackHole.pos.x = centerX;
    blackHole.pos.y = centerY;
    blackHole.vel.x = 0;
    blackHole.vel.y = 0;
    blackHole.mass = coreMass;
    blackHole.colorVal = 255;
    particles.push_back(blackHole);

    // 2. Bintang-bintang
    for (int i = 1; i < NUM_PARTICLES; i++)
    {
        Particle p;

        // --- LOGIKA DISK (PIRINGAN) ---
        // Jarak Random: Kita pakai akar(random) agar sebarannya merata tapi tetap padat di tengah
        // Kalau pakai random biasa, nanti tengahnya terlalu kosong.
        float r = maxDist * std::sqrt(randomFloat());
        if (r < 15.0f)
            r = 15.0f; // Jangan terlalu dekat black hole

        // Sudut Random (0 - 360 derajat bebas)
        float theta = randomFloat() * 6.28318f;

        // Posisi Cartesian
        p.pos.x = centerX + r * std::cos(theta);
        p.pos.y = centerY + r * std::sin(theta);

        // --- KECEPATAN ORBIT (KEPLER) ---
        // V = sqrt(G * M / r)
        float v = std::sqrt(0.5f * coreMass / r);

        // Variasi Kecepatan (PENTING!)
        // Kita acak sedikit (0.8x sampai 1.2x) biar orbitnya lonjong-lonjong dikit (elips)
        // Ini yang bikin galaksinya terlihat "hidup", bukan kayak piringan kaset yang kaku.
        float variation = 0.8f + (randomFloat() * 0.4f);
        v *= variation;

        // Arah Tegak Lurus (Tangensial)
        p.vel.x = -std::sin(theta) * v;
        p.vel.y = std::cos(theta) * v;

        p.mass = randomFloat() * 2.0f + 0.5f;
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
        }

        sf::Vector2i pixelPos = sf::Mouse::getPosition(window);

        sf::Vector2f worldPos = window.mapPixelToCoords(pixelPos);

        bool isPressed = sf::Mouse::isButtonPressed(sf::Mouse::Left);

        // hitung fisika (GPU)
        launchCudaBody(d_particles, NUM_PARTICLES, blockPerGrid, threadsPerBlock, worldPos.x, worldPos.y, isPressed);

        // copy data gpu ke cpu
        copyFromDevice(host_particles.data(), d_particles, size);

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
                // LAMBAT (Pinggir) -> UNGU GELAP
                // R tinggi, G rendah, B tinggi
                r = 80 + (int)(speed * 20);
                g = 0;
                b = 200;
                a = 30; // Sangat transparan (biar kalau numpuk jadi terang)
            }
            else if (speed < 12.0f)
            {
                // SEDANG (Orbit Stabil) -> CYAN / BIRU MUDA
                // R rendah, G tinggi, B tinggi
                r = 0;
                g = 150 + (int)(speed * 5);
                b = 255;
                a = 60;
            }
            else
            {
                // NGEBUT (Dekat Blackhole) -> PUTIH MENYALA
                r = 255;
                g = 255;
                b = 255;
                a = 150; // Lebih solid
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
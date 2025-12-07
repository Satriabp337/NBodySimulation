#include "physics.h"
#include <cmath> // Untuk std::sqrt
#include <omp.h>

void cpuBodyInteraction(Particle* particles, int n) {
    // Loop untuk setiap partikel (i)
    for (int i = 0; i < n; i++) {
        float fx = 0.0f;
        float fy = 0.0f;

        // Loop interaksi dengan semua partikel lain (j) -> O(N^2)
        for (int j = 0; j < n; j++) {
            if (i == j) continue; // Jangan hitung diri sendiri

            // Hitung jarak vektor
            float dx = particles[j].pos.x - particles[i].pos.x;
            float dy = particles[j].pos.y - particles[i].pos.y;

            // Hitung jarak kuadrat + Softening (Biar gak meledak)
            float distSq = dx*dx + dy*dy + SOFTENING * SOFTENING;
            float dist = std::sqrt(distSq);

            // Rumus Gravitasi (Sama persis dengan Kernel CUDA)
            // F = G * m / r^3 * vector_r
            // (Kita pakai r^3 di penyebut karena dikali dx/dy di pembilang)
            float f = (G_CONST * particles[j].mass) / (distSq * dist);

            fx += f * dx;
            fy += f * dy;
        }

        // Update Kecepatan (Velocity)
        particles[i].vel.x += fx * DT;
        particles[i].vel.y += fy * DT;

        // Update Posisi (Position)
        particles[i].pos.x += particles[i].vel.x * DT;
        particles[i].pos.y += particles[i].vel.y * DT;
    }
}

void cpuBodyInteractionOpenMP(Particle* particles, int n) {
    
    // GUNAKAN SCHEDULE(STATIC)
    // Static membagi tugas di awal secara rata (misal: 0-1000 ke Core 1, 1001-2000 ke Core 2).
    // Ini jauh lebih cepat untuk N-Body karena beban tiap partikel sama persis.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float fx = 0.0f;
        float fy = 0.0f;

        for (int j = 0; j < n; j++) {
            if (i == j) continue;

            float dx = particles[j].pos.x - particles[i].pos.x;
            float dy = particles[j].pos.y - particles[i].pos.y;
            float distSq = dx*dx + dy*dy + SOFTENING * SOFTENING;
            float dist = std::sqrt(distSq);

            float f = (G_CONST * particles[j].mass) / (distSq * dist);

            fx += f * dx;
            fy += f * dy;
        }

        particles[i].vel.x += fx * DT;
        particles[i].vel.y += fy * DT;

        particles[i].pos.x += particles[i].vel.x * DT;
        particles[i].pos.y += particles[i].vel.y * DT;
    }
}

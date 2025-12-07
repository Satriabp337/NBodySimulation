#include "physics.h"
#include <cmath>

void cpuBodyInteraction(Particle* particles, int n) {
    for (int i = 0; i < n; i++) {
        float fx=0, fy=0;

        for (int j = 0; j < n; j++) {
            if (i==j) continue;

            float dx = particles[j].pos.x - particles[i].pos.x;
            float dy = particles[j].pos.y - particles[i].pos.y;

            float distSq = dx*dx + dy*dy + SOFTENING*SOFTENING;
            float dist = std::sqrt(distSq);
            float force = (G_CONST * particles[j].mass) / (distSq * dist);

            fx += force*dx;
            fy += force*dy;
        }

        particles[i].vel.x += fx*DT;
        particles[i].vel.y += fy*DT;

        particles[i].pos.x += particles[i].vel.x*DT;
        particles[i].pos.y += particles[i].vel.y*DT;
    }
}

#include <SFML/Graphics.hpp>
#include <iostream>

// Deklarasi fungsi CUDA (nanti isinya di kernel.cu)
void printCudaInfo(); 

int main() {
    // 1. Cek CUDA
    std::cout << "Mengecek CUDA..." << std::endl;
    printCudaInfo();

    // 2. Cek SFML
    sf::RenderWindow window(sf::VideoMode(800, 600), "N-Body Simulation (CUDA + SFML)");
    sf::CircleShape shape(100.f);
    shape.setFillColor(sf::Color::Green);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        window.draw(shape);
        window.display();
    }

    return 0;
}
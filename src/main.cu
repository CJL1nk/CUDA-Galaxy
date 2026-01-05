#include "includes.h"
#include "kernel.cuh"

#define N 53000000

// 1. Pass all bodies and density map in to GPU
// 2. Have GPU calculate the next position, velocity, and acceleration of every body. Update density map
//    (Density map is a map of how many bodies are in each pixel, as each pixel on screen can hold multiple bodies and will be brighter the more bodies are in that pixel)
// 4. GPU Passes density map back to cpu
// 5. CPU Loops through density map and draws pixels on screen, according to how many bodies are in each pixel

int main() {

    // Setup +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // SFML Window stuff
    sf::Vector2u windowSize = sf::Vector2u(1920, 1080);
    sf::RenderWindow window(sf::VideoMode(windowSize), "Big Thingy", sf::Style::Default);
    window.setFramerateLimit(60);

    constexpr int numBodies = 10000000;

    // +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=

    // Map sequentially containing all bodies
    Body* bodies = (Body*)malloc(numBodies * sizeof(Body));

    // Map containing the number of bodies per pixel
    int* densityMap = (int*)calloc((windowSize.x * windowSize.y), sizeof(int));

    std::uniform_real_distribution<double> randomX(0.0, (double)windowSize.x * 1000.0);
    std::uniform_real_distribution<double> randomY(0.0, (double)windowSize.y * 1000.0);
    std::uniform_real_distribution<double> randomMass(0.0, 10000.0);

    sf::VertexArray points(sf::PrimitiveType::Points);

    for (int i = 0; i < numBodies; i++) {

        double xPos = randomX(gen);
        double yPos = randomY(gen);

        int xCoordinate = xPos / 1000;
        int yCoordinate = yPos / 1000;

        bodies[i] = (Body){Position{xPos, yPos, 0, 0, 0, 0}, randomMass(gen)};


        densityMap[(yCoordinate * windowSize.x + xCoordinate)] += 1;

        // std::cout << densityMap[(yCoordinate * windowSize.x + xCoordinate)] << " Bodies at " << xCoordinate << ", " << yCoordinate << std::endl;

        points.append(sf::Vertex{
                    sf::Vector2f(xCoordinate, yCoordinate),
                            sf::Color(densityMap[yCoordinate * windowSize.x + xCoordinate] * 20,
                                    densityMap[yCoordinate * windowSize.x + xCoordinate] * 20,
                                     densityMap[yCoordinate * windowSize.x + xCoordinate] * 20)});
    }

    std::cout << "done" << std::endl;

    window.draw(points);
    window.display();

    while (window.isOpen()) {
        if (isKeyPressed(sf::Keyboard::Key::Escape)) {
            window.close();
        }
    }

    // Make device copy of bodies and density map
    Body* d_bodies;
    int* d_densityMap;

    int bodiesSize = numBodies * sizeof(Body);
    int mapSize = windowSize.x * windowSize.y * sizeof(int);

    cudaMalloc((void**)&d_bodies, bodiesSize);
    cudaMalloc((void**)&d_densityMap, mapSize);

    updateBodyPositions<<<1, 1>>>(d_bodies, bodiesSize, d_densityMap, mapSize);
    /*
    // Host copies
    int* a;
    int* b;
    int* c;

    // Device copies
    int* d_a;
    int* d_b;
    int* d_c;

    constexpr int size = N * sizeof(int);

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i;
    }

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    skibidiKernel<<<N/512, 512 >>>(d_a, d_b, d_c, N);
    auto stop = std::chrono::high_resolution_clock::now();

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    auto duration = (stop - start);
    std::cout << duration.count() << " on GPU" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
    stop = std::chrono::high_resolution_clock::now();
    duration = (stop - start);
    std::cout << duration.count() << " on CPU" << std::endl;

    for (int i = 0; i < N; i++) {
        std::cout << a[i] << " " << b[i] << " " << c[i] << std::endl;
    }

    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: "
                  << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "Kernel executed successfully." << std::endl; */
    return 0;
}
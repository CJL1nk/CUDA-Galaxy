#include "includes.h"
#include "kernel.cuh"

#define N 53000000

int main() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distrib(1, 100);

    sf::Vector2u windowSize = sf::Vector2u(480, 480);
    sf::RenderWindow window(sf::VideoMode(windowSize), "Particle Simulation");
    window.setFramerateLimit(60);

    window.clear(sf::Color::Black);
    window.display();

    constexpr int maxBodies = 2000000;
    int bodyCount = 0;

    sf::Vector2i center = sf::Vector2i(windowSize.x / 2, windowSize.y / 2);

    while (bodyCount < maxBodies) {

        std::uniform_int_distribution<int> distrib(1, 480);

        int xPos = distrib(gen);
        int yPos = distrib(gen);

        int distFromCenter = (int)sqrt(pow(fabs(xPos - center.x), 2) + pow(fabs(yPos - center.y), 2)) + 1;
        int bodiesToGenerate = (int)sqrt(pow(fabs(windowSize.x - center.x), 2) + pow(fabs(windowSize.y - center.y), 2)) - distFromCenter;

        std::cout << "Generating " << bodiesToGenerate << " at " << xPos << ", " << yPos << std::endl;

        sf::Vertex point(sf::Vector2f(xPos, yPos), sf::Color(0, (uint8_t)bodiesToGenerate, (uint8_t)bodiesToGenerate >> 8));
        window.draw(&point, 1, sf::PrimitiveType::Points);

        bodyCount+= bodiesToGenerate;
    }

    std::cout << bodyCount << std::endl;
    window.display();

    while (true) {

    }

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

    /*for (int i = 0; i < N; i++) {
        std::cout << a[i] << " " << b[i] << " " << c[i] << std::endl;
    }*/

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

    std::cout << "Kernel executed successfully." << std::endl;
    return 0;
}
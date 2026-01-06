#include "helpers.h"
#include "includes.h"
#include "kernel.cuh"

int main() {

    // Setup +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=

    // SFML Window stuff
    sf::Vector2u windowSize = sf::Vector2u(1000, 1000);
    sf::RenderWindow window(sf::VideoMode(windowSize), "CUDA Galaxy", sf::Style::Default);
    window.setFramerateLimit(60);
    sf::VertexArray points(sf::PrimitiveType::Points);

    float distancePerPixel = 10.0f;

    constexpr int numBodies = 1500000;
    int bodiesSize = numBodies * sizeof(float);

    float* bodyXPos = (float*)malloc(bodiesSize);
    float* bodyYPos = (float*)malloc(bodiesSize);
    float* mass = (float*)malloc(bodiesSize);

    // +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=


    generateRandomBodies(numBodies, bodyXPos, bodyYPos, mass, windowSize, distancePerPixel);
    std::cout << "Bodies generated" << std::endl;

    int numCells = windowSize.x * windowSize.y;

    // Map containing how many bodies are in each pixel
    int* densityMap = (int*)malloc(numCells * sizeof(int));

    // CUDA Setup -----------------------------------------------------------------------------------

    // Device copy of body positions
    float* d_bodyXPos;
    float* d_bodyYPos;
    float* d_mass;

    // GPU only, velocity of each body
    float* d_bodyXVel;
    float* d_bodyYVel;

    // Map containing which cell each body lays in
    int* d_bodyCells;

    int* d_densityMap;

    cudaMalloc((void**)&d_bodyXPos, bodiesSize);
    cudaMalloc((void**)&d_bodyYPos, bodiesSize);
    cudaMalloc((void**)&d_bodyXVel, bodiesSize);
    cudaMalloc((void**)&d_bodyYVel, bodiesSize);
    cudaMalloc((void**)&d_mass, bodiesSize);
    cudaMalloc((void**)&d_bodyCells, numBodies * sizeof(int));
    cudaMalloc((void**)&d_densityMap, numCells * sizeof(int));

    cudaMemcpy(d_bodyXPos, bodyXPos, bodiesSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bodyYPos, bodyYPos, bodiesSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, numBodies * sizeof(float), cudaMemcpyHostToDevice);

    // Free host memory, as CPU will only read from density map
    free(bodyXPos);
    free(bodyYPos);
    free(mass);

    // ----------------------------------------------------------------------------------------------

    int threadsPerBlock = 256;
    int blocks = (numBodies + threadsPerBlock - 1) / threadsPerBlock;

    // Main Loop
    while (window.isOpen()) {

        clearDensityMap<<<blocks, threadsPerBlock>>>(d_densityMap, numCells);

        gridBodies<<<blocks, threadsPerBlock>>>(d_bodyXPos, d_bodyYPos, d_bodyCells, numBodies, distancePerPixel, windowSize.x);

        generateDensityMap<<<blocks, threadsPerBlock>>>(d_bodyCells, d_densityMap, numBodies);

        cudaDeviceSynchronize();
        cudaMemcpy(densityMap, d_densityMap, numCells * sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < numCells; i++) {

            int x = i % windowSize.x;
            int y = i / windowSize.x;

            points.append(sf::Vertex{
                   sf::Vector2f(x, y),
                           sf::Color(densityMap[y * windowSize.x + x] * 20,
                                   densityMap[y * windowSize.x + x] * 20,
                                    densityMap[y * windowSize.x + x] * 20)});
        }

        window.draw(points);
        window.display();

        if (isKeyPressed(sf::Keyboard::Key::Escape)) {
            window.close();
        }
    }

    cudaDeviceSynchronize();
    cudaMemcpy(densityMap, d_densityMap, numCells * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_bodyXPos);
    cudaFree(d_bodyYPos);
    cudaFree(d_bodyXVel);
    cudaFree(d_bodyYVel);
    cudaFree(d_mass);
    cudaFree(d_densityMap);
    cudaFree(d_bodyCells);

    return 0;
}
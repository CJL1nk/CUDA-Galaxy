#include "helpers.h"
#include "includes.h"
#include "gl.h"
#include "kernel.cuh"

int main() {

    // GLFW Setup +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+

    // GLFW Window stuff
    const WindowSize windowSize = {1920, 1080}; // Going above ~1250x1200 for some reason breaks the program
    GLFWwindow* window = createWindow(windowSize.width, windowSize.height, "CUDA Galaxy");

    // Fullscreen
    const float vertices[] = {
        -1.f, -1.f,   0.f, 0.f,
         1.f, -1.f,   1.f, 0.f,
         1.f,  1.f,   1.f, 1.f,
        -1.f,  1.f,   0.f, 1.f
    };
    const unsigned int indices[] = {0, 1, 2, 2, 3, 0};

    GLuint VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    GLuint shaderProgram = createShaderProgram();

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, windowSize.width, windowSize.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // End of OpenGL setup

    float distancePerPixel = 50.0f;

    constexpr int numBodies = 150000;
    int bodiesSize = numBodies * sizeof(float);
    int numCells = windowSize.width * windowSize.height;

    float* bodyXPos = (float*)malloc(bodiesSize);
    float* bodyYPos = (float*)malloc(bodiesSize);
    float* bodyXVel = (float*)malloc(bodiesSize);
    float* bodyYVel = (float*)malloc(bodiesSize);
    float* mass = (float*)malloc(bodiesSize);

    // Raw pixel stream to be rendered
    uint8_t* pixelStream = new uint8_t[numCells * 4];

    // +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=

    //generateRandomBodies(numBodies, bodyXPos, bodyYPos, mass, windowSize, distancePerPixel);
    generateGalaxyBodies(numBodies, bodyXPos, bodyYPos, bodyXVel, bodyYVel, mass, windowSize, distancePerPixel);
    std::cout << "Bodies generated" << std::endl;

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

    // Map containing how many pixels are in each cell
    int* d_densityMap;

    // Pixel output stream (RGBA uint8)
    uint8_t* d_heatMap;

    cudaMalloc((void**)&d_bodyXPos, bodiesSize);
    cudaMalloc((void**)&d_bodyYPos, bodiesSize);
    cudaMalloc((void**)&d_bodyXVel, bodiesSize);
    cudaMalloc((void**)&d_bodyYVel, bodiesSize);
    cudaMalloc((void**)&d_mass, bodiesSize);
    cudaMalloc((void**)&d_bodyCells, numBodies * sizeof(int));
    cudaMalloc((void**)&d_densityMap, numCells * sizeof(int));
    cudaMalloc(&d_heatMap, numCells * 4 * sizeof(uint8_t));

    cudaMemcpy(d_bodyXPos, bodyXPos, bodiesSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bodyYPos, bodyYPos, bodiesSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bodyXVel, bodyXVel, bodiesSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bodyYVel, bodyYVel, bodiesSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, numBodies * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_bodyXVel, 0, bodiesSize);
    cudaMemset(d_bodyYVel, 0, bodiesSize);

    // Free host memory, as CPU will only read from heat map
    free(bodyXPos);
    free(bodyYPos);
    free(bodyXVel);
    free(bodyYVel);
    free(mass);

    constexpr int threadsPerBlock = 512;
    int blocks = (numBodies + threadsPerBlock - 1) / threadsPerBlock;
    int cellBlocks = (numCells + threadsPerBlock - 1) / threadsPerBlock;

    // ----------------------------------------------------------------------------------------------

    // Main loop, no input handling for 0.0002 less CPU cycles per year
    while (!glfwWindowShouldClose(window)) {

        glfwPollEvents();

        clearDensityMap<<<cellBlocks, threadsPerBlock>>>(d_densityMap, d_heatMap, numCells);

        gridBodies<<<blocks, threadsPerBlock>>>(d_bodyXPos, d_bodyYPos, d_bodyCells, numBodies, distancePerPixel, windowSize.width, windowSize.height);

        // Shared mem for W speed W kai cenat
        size_t shmemSize = 3 * threadsPerBlock * sizeof(float);
        physicsKernel<<<blocks, threadsPerBlock, shmemSize>>>(d_bodyXPos, d_bodyYPos, d_bodyXVel, d_bodyYVel, d_mass, d_bodyCells, numBodies);
        positionKernel<<<blocks, threadsPerBlock>>>(d_bodyXPos, d_bodyYPos, d_bodyXVel, d_bodyYVel, numBodies);

        generateDensityMap<<<blocks, threadsPerBlock>>>(d_bodyCells, d_densityMap, numBodies);
        generateHeatMap<<<cellBlocks, threadsPerBlock>>>(d_densityMap, d_heatMap, numCells);

        cudaDeviceSynchronize();
        cudaMemcpy(pixelStream, d_heatMap, numCells * 4 * sizeof(uint8_t), cudaMemcpyDeviceToHost); // Super RIP performance btw

        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, windowSize.width, windowSize.height, GL_RGBA, GL_UNSIGNED_BYTE, pixelStream);

        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glBindTexture(GL_TEXTURE_2D, texture);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteTextures(1, &texture);

    glfwDestroyWindow(window);
    glfwTerminate();

    cudaFree(d_bodyXPos);
    cudaFree(d_bodyYPos);
    cudaFree(d_bodyXVel);
    cudaFree(d_bodyYVel);
    cudaFree(d_mass);
    cudaFree(d_densityMap);
    cudaFree(d_bodyCells);

    return 0;
}
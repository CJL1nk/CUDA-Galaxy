#include "helpers.h"
#include "includes.h"
#include "gl.h"
#include "kernel.cuh"

int main() {

    // GLFW Setup +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+

    // GLFW Window stuff
    WindowSize windowSize = {1920, 1080};
    GLFWwindow* window = createWindow(windowSize.width, windowSize.height, "CUDA Galaxy");

    // Fullscreen quad
    float vertices[] = {
        // positions   // tex coords
        -1.f, -1.f,   0.f, 0.f,
         1.f, -1.f,   1.f, 0.f,
         1.f,  1.f,   1.f, 1.f,
        -1.f,  1.f,   0.f, 1.f
    };
    unsigned int indices[] = {0, 1, 2, 2, 3, 0};

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

    float distancePerPixel = 10.0f;

    constexpr int numBodies = 1500000;
    int bodiesSize = numBodies * sizeof(float);
    int numCells = windowSize.width * windowSize.height;

    float* bodyXPos = (float*)malloc(bodiesSize);
    float* bodyYPos = (float*)malloc(bodiesSize);
    float* mass = (float*)malloc(bodiesSize);

    // Raw pixel stream to be rendered
    uint8_t pixelStream[windowSize.width * windowSize.height * 4];

    // Map containing how many bodies are in each pixel
    int* densityMap = (int*)malloc(numCells * sizeof(int));

    // +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=

    generateRandomBodies(numBodies, bodyXPos, bodyYPos, mass, windowSize, distancePerPixel);
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

    int threadsPerBlock = 512;
    int blocks = (numBodies + threadsPerBlock - 1) / threadsPerBlock;

    clearDensityMap<<<blocks, threadsPerBlock>>>(d_densityMap, numCells);

    gridBodies<<<blocks, threadsPerBlock>>>(d_bodyXPos, d_bodyYPos, d_bodyCells, numBodies, distancePerPixel, windowSize.width);

    generateDensityMap<<<blocks, threadsPerBlock>>>(d_bodyCells, d_densityMap, numBodies);

    cudaDeviceSynchronize();
    cudaMemcpy(densityMap, d_densityMap, numCells * sizeof(int), cudaMemcpyDeviceToHost);


    while (!glfwWindowShouldClose(window)) {

        glfwPollEvents();

        clearDensityMap<<<blocks, threadsPerBlock>>>(d_densityMap, numCells);

        gridBodies<<<blocks, threadsPerBlock>>>(d_bodyXPos, d_bodyYPos, d_bodyCells, numBodies, distancePerPixel, windowSize.width);

        generateDensityMap<<<blocks, threadsPerBlock>>>(d_bodyCells, d_densityMap, numBodies);

        cudaDeviceSynchronize();
        cudaMemcpy(pixelStream, d_densityMap, numCells * sizeof(int), cudaMemcpyDeviceToHost);

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
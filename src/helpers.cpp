//
// Created by CJ on 1/5/26.
//

#include "helpers.h"

void generateRandomBodies(int numBodies, float* bodyXPos, float* bodyYPos, float* mass, sf::Vector2u windowSize, float distancePerPixel) {

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> randomX(0.0, static_cast<float>(windowSize.x) * distancePerPixel);
    std::uniform_real_distribution<float> randomY(0.0, static_cast<float>(windowSize.y) * distancePerPixel);
    std::uniform_real_distribution<float> randomMass(0.0, 10000.0);

    for (int i = 0; i < numBodies; i++) {
        bodyXPos[i] = randomX(gen);
        bodyYPos[i] = randomY(gen);
        mass[i] = randomMass(gen);
    }
}
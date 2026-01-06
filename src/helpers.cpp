//
// Created by CJ on 1/5/26.
//

#include "helpers.h"

void generateRandomBodies(int numBodies, float* bodyXPos, float* bodyYPos, float* mass, WindowSize windowSize, float distancePerPixel) {

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> randomX(0.0, static_cast<float>(windowSize.width) * distancePerPixel);
    std::uniform_real_distribution<float> randomY(0.0, static_cast<float>(windowSize.height) * distancePerPixel);
    std::uniform_real_distribution<float> randomMass(150.0, 2000.0);

    for (int i = 0; i < numBodies; i++) {
        bodyXPos[i] = randomX(gen);
        bodyYPos[i] = randomY(gen);
        mass[i] = randomMass(gen);
    }
}

// This function was completely AI generated I just needed something real quick for testing purposes
void generateGalaxyBodies(int numBodies, float* bodyXPos, float* bodyYPos, float* bodyXVel, float* bodyYVel, float* mass,
                         WindowSize windowSize, float distancePerPixel) {

    std::random_device rd;
    std::mt19937 gen(rd());

    float centerX = (windowSize.width * distancePerPixel) / 2.0f;
    float centerY = (windowSize.height * distancePerPixel) / 2.0f;
    float maxRadius = std::min(windowSize.width, windowSize.height) * distancePerPixel * 0.45f;

    std::uniform_real_distribution<float> randomAngle(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> randomMass(150.0f, 300.0f);
    std::normal_distribution<float> randomNoise(0.0f, maxRadius * 0.05f);
    std::exponential_distribution<float> radialDist(2.0f / maxRadius);

    for (int i = 0; i < numBodies; i++) {
        float radius = radialDist(gen);
        if (radius > maxRadius) radius = maxRadius;

        float angle = randomAngle(gen);
        float spiralTightness = 3.0f;
        float spiralAngle = angle + (radius / maxRadius) * spiralTightness * 2.0f * M_PI;

        std::uniform_real_distribution<float> armChance(0.0f, 1.0f);
        bool inArm = armChance(gen) < 0.6f;

        float finalAngle;
        if (inArm) {
            int armIndex = (i % 2);
            float armOffset = armIndex * M_PI;
            finalAngle = spiralAngle + armOffset;
            std::normal_distribution<float> armThickness(0.0f, 0.3f);
            finalAngle += armThickness(gen);
        } else {
            finalAngle = angle;
        }

        float x = centerX + radius * std::cos(finalAngle);
        float y = centerY + radius * std::sin(finalAngle);

        x += randomNoise(gen);
        y += randomNoise(gen);

        bodyXPos[i] = x;
        bodyYPos[i] = y;

        // **ADD ORBITAL VELOCITY** - This is crucial!
        float dx = x - centerX;
        float dy = y - centerY;
        float r = std::sqrt(dx * dx + dy * dy);

        if (r > 0.01f) {
            // Keplerian velocity: v = sqrt(GM/r)
            // For visual effect, use a simplified version
            float orbitalSpeed = std::sqrt(10000.0f / (r + 10.0f)) * 0.8f;

            // Perpendicular to radius (tangential velocity)
            bodyXVel[i] = -dy / r * orbitalSpeed;
            bodyYVel[i] = dx / r * orbitalSpeed;

            // Add small random component
            std::normal_distribution<float> velNoise(0.0f, orbitalSpeed * 0.1f);
            bodyXVel[i] += velNoise(gen);
            bodyYVel[i] += velNoise(gen);
        } else {
            bodyXVel[i] = 0.0f;
            bodyYVel[i] = 0.0f;
        }

        float massFactor = 1.0f + (1.0f - radius / maxRadius) * 2.0f;
        mass[i] = randomMass(gen) * massFactor;
    }
}
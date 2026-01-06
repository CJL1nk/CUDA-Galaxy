//
// Created by CJ on 1/5/26.
//

#ifndef HELPERS_H
#define HELPERS_H

#include "includes.h"

/**
 * Populates position arrays with bodies with randomized positions and mass
 * N'th spot in an array represents that data for the N'th body
 * @param numBodies Number of bodies to generate
 * @param bodyXPos Array containing all body x positions
 * @param bodyYPos Array containing all body y positions
 * @param mass Array containing all body masses
 * @param windowSize Dimensions of rendering window
 * @param distancePerPixel Amount of "simulation distance" per pixel (e.g. 1000.f for 1000 "miles" per pixel)
 */
void generateRandomBodies(int numBodies, float* bodyXPos, float* bodyYPos, float* mass, WindowSize windowSize, float distancePerPixel);

#endif //HELPERS_H

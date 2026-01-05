//
// Created by CJ on 1/4/26.
//

#ifndef TYPES_H
#define TYPES_H

struct Position {
    double xPos;
    double yPos;

    double accelX;
    double accelY;

    double velX;
    double velY;
};

struct Body {
    Position position;
    double mass;
};

#endif //TYPES_H

//
// Created by CJ on 1/5/26.
//

#ifndef GL_H
#define GL_H

#include "includes.h"

GLuint compileShader(GLenum type, const char* src);

GLFWwindow* createWindow(int width, int height, const char* title);

GLuint createShaderProgram();

#endif //GL_H

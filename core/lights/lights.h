//
// Created by 孙万捷 on 16/5/25.
//

#ifndef SUNVOLUMERENDER_LIGHTS_H
#define SUNVOLUMERENDER_LIGHTS_H

#include <iostream>
#include <vector>
#include <string>

#include <glm/glm.hpp>

#include "cuda_environment_light.h"

class Lights
{
public:
    Lights();
    ~Lights();
    void SetEnvironmentLight(std::string filename);
    void SetEnvionmentLight(const glm::vec3& radiance);

public:
    cudaArray* envMapArray = nullptr;
    cudaEnvironmentLight environmentLight;
};

#endif //SUNVOLUMERENDER_LIGHTS_H

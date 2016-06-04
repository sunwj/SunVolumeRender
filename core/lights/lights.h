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
#include "cuda_arealight.h"
#include "../../common.h"

class Lights
{
public:
    Lights();
    ~Lights();
    void SetEnvironmentLight(std::string filename);
    void SetEnvionmentLight(const glm::vec3& radiance);
    void SetEnvironmentLightIntensity(float intensity);
    void SetEnvironmentLightOffset(const glm::vec2& offset);

    void AddAreaLights(const cudaAreaLight& areaLight, const glm::vec3& tm);
    void RemoveLights(uint32_t idx);
    cudaAreaLight* GetAreaLight(uint32_t idx);
    cudaAreaLight* GetLastAreaLight();
    glm::vec3* GetAreaLightTransformation(uint32_t idx);


public:
    cudaArray* envMapArray = nullptr;
    cudaEnvironmentLight environmentLight;

    std::vector<cudaAreaLight> areaLights;
    std::vector<glm::vec3> transforms;      // x and y represents latitude and longitude, z is distance
};

#endif //SUNVOLUMERENDER_LIGHTS_H

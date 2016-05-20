//
// Created by 孙万捷 on 16/5/20.
//

#ifndef SUNVOLUMERENDER_SCATTER_EVENT_H
#define SUNVOLUMERENDER_SCATTER_EVENT_H

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include <cuda_runtime.h>

class ScatterEvent
{
public:
    float intensity = 0.f;
    glm::vec3 pointInWorld = glm::vec3(glm::uninitialize);
    glm::vec3 normalizedGradient = glm::vec3(glm::uninitialize);
    float gradientMagnitude = 0.f;
};

#endif //SUNVOLUMERENDER_SCATTER_EVENT_H

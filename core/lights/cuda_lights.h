//
// Created by 孙万捷 on 16/5/25.
//

#ifndef SUNVOLUMERENDER_CUDALIGHTS_H
#define SUNVOLUMERENDER_CUDALIGHTS_H

#include <cuda_runtime.h>

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include "cuda_environment_light.h"

class cudaLights
{
public:
    __device__ glm::vec3 GetEnvironmentRadiance(const glm::vec3& dir, float u_offset = 0.f, float v_offset = 0.f)
    {
        return environmentLight.GetEnvRadiance(dir, u_offset, v_offset);
    }

public:
    cudaEnvironmentLight environmentLight;
};

#endif //SUNVOLUMERENDER_CUDALIGHTS_H

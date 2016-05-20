//
// Created by 孙万捷 on 16/2/6.
//

#ifndef SUNPATHTRACER_RAY_H
#define SUNPATHTRACER_RAY_H

#include <float.h>

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include <cuda_runtime.h>

class cudaRay
{
public:
    __device__ cudaRay()
    {
        tMin = 1e-8;
        tMax = FLT_MAX;
    }

    __device__ cudaRay(const glm::vec3& orig, const glm::vec3& dir, float tMin = 1e-8, float tMax = FLT_MAX)
    {
        this->orig = orig;
        this->dir = dir;
        this->tMin = tMin;
        this->tMax = tMax;
    }

    __device__ glm::vec3 PointOnRay(float t) const
    {
        return orig + t * dir;
    }

public:
    glm::vec3 orig;
    glm::vec3 dir;
    mutable float tMin;
    mutable float tMax;
};

#endif //SUNPATHTRACER_RAY_H

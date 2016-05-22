//
// Created by 孙万捷 on 16/5/22.
//

#ifndef SUNVOLUMERENDER_CUDA_DISK_H
#define SUNVOLUMERENDER_CUDA_DISK_H

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include <cuda_runtime.h>

#include "../cuda_ray.h"

class cudaDisk
{
public:
    __device__ bool Intersect(const cudaRay& ray, float* t)
    {
        auto denom = glm::dot(normal, ray.dir);
        if(denom > 1e-6)
        {
            auto co = center - ray.orig;
            *t = glm::dot(co, normal) / denom;

            if(*t >= 0)
            {
                auto p = ray.orig + *t * ray.dir;
                auto co = p - center;
                return sqrtf(glm::length(v) <= radius);
            }

            return false;
        }

        return false;
    }

private:
    float radius;
    glm::vec3 center;
    glm::vec3 normal;
};

#endif //SUNVOLUMERENDER_CUDA_DISK_H

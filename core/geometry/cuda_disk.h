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
    __host__ __device__ cudaDisk() {}

    __host__ __device__ cudaDisk(const glm::vec3& center, const glm::vec3& normal, float radius)
    {
        Set(center, normal, radius);
    }

    __host__ __device__ void Set(const glm::vec3& center, const glm::vec3& normal, float radius)
    {
        this->center = center;
        this->normal = normal;
        this->radius = radius;
    }

    __device__ bool Intersect(const cudaRay& ray, float* t) const
    {
        auto denom = glm::dot(normal, ray.dir);
        if(fabsf(denom) > 1e-6)
        {
            auto co = center - ray.orig;
            *t = glm::dot(co, normal) / denom;

            if(*t >= 0)
            {
                auto p = ray.orig + *t * ray.dir;
                auto co = p - center;
                return sqrtf(glm::length(co)) <= radius;
            }

            return false;
        }

        return false;
    }

    __host__ __device__ float GetArea() const
    {
        return M_PI * radius * radius;
    }

public:
    float radius;
    glm::vec3 center;
    glm::vec3 normal;
};

#endif //SUNVOLUMERENDER_CUDA_DISK_H

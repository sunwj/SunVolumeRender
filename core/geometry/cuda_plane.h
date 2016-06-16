//
// Created by 孙万捷 on 16/6/15.
//

#ifndef SUNVOLUMERENDER_CUDA_PLANE_H
#define SUNVOLUMERENDER_CUDA_PLANE_H

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include <cuda_runtime.h>

#include "../cuda_ray.h"

class cudaPlane
{
public:
    __host__ __device__ cudaPlane() {}

    __host__ __device__ cudaPlane(const glm::vec3& p, const glm::vec3& normal)
    {
        Set(p, normal);
    }

    __host__ __device__ void Set(const glm::vec3& _p, const glm::vec3& _normal)
    {
        p = _p;
        normal = _normal;
    }

    __host__ __device__ bool Intersect(const cudaRay& ray, float* t) const
    {
        auto denom = glm::dot(normal, ray.dir);
        if(fabsf(denom) > 1e-6)
        {
            auto po = p - ray.orig;
            *t = glm::dot(po, normal) / denom;

            return *t > 0.f;
        }

        return false;
    }

public:
    glm::vec3 p;
    glm::vec3 normal;
};

#endif //SUNVOLUMERENDER_CUDA_PLANE_H

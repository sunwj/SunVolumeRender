//
// Created by 孙万捷 on 16/5/26.
//

#ifndef SUNVOLUMERENDER_CUDA_LIGHT_H
#define SUNVOLUMERENDER_CUDA_LIGHT_H

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include <cuda_runtime.h>

#include "../geometry/cuda_disk.h"

class cudaAreaLight
{
public:
    __host__ void Set(const cudaDisk& disk, const glm::vec3& color, float intensity)
    {
        this->disk = disk;
        this->color = color;
        this->intensity = intensity;
    }

    __host__ void SetShape(const cudaDisk& disk)
    {
        this->disk = disk;
    }

    __host__ void SetColor(const glm::vec3& color)
    {
        this->color = color;
    }

    __host__ void SetIntensity(float intensity)
    {
        this->intensity = intensity;
    }

    __host__ void SetRadius(float radius)
    {
        this->disk.radius = radius;
    }

    __host__ void SetPosition(const glm::vec3& pos)
    {
        this->disk.center = pos;
    }

    __host__ void SetNormal(const glm::vec3& normal)
    {
        this->disk.normal = normal;
    }

    __host__ __device__ glm::vec3 GetColor() const {return color;}
    __host__ __device__ float GetIntensity() const {return intensity;}
    __host__ __device__ glm::vec3 GetRadiance() const {return color * intensity / disk.GetArea();}
    __host__ __device__ float GetRadius() const {return disk.radius;}
    __host__ __device__ float GetArea() const {return disk.GetArea();}
    __host__ __device__ glm::vec3 GetCenter() const {return disk.center;}
    __host__ __device__ glm::vec3 GetNormal(const glm::vec3& pt) const {return disk.normal;}

    __device__ bool Intersect(const cudaRay& ray, float* t)
    {
        return disk.Intersect(ray, t);
    }

private:
    cudaDisk disk;
    glm::vec3 color;
    float intensity;
};

#endif //SUNVOLUMERENDER_CUDA_LIGHT_H

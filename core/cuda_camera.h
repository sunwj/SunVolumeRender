//
// Created by 孙万捷 on 16/2/6.
//

#ifndef SUNPATHTRACER_CAMERA_H
#define SUNPATHTRACER_CAMERA_H

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include "cuda_ray.h"

class cudaCamera
{
public:
    __host__ __device__ cudaCamera() {}

    __host__ __device__ cudaCamera(const glm::vec3& _pos, const glm::vec3& _u, const glm::vec3& _v, const glm::vec3& _w, float fovx = 45.f, unsigned int _imageW = 640, unsigned int _imageH = 480)
    {
        Setup(_pos, _u, _v, _w, fovx, _imageW, _imageH);
    }

    __host__ __device__ cudaCamera(const glm::vec3& _pos, const glm::vec3& target, const glm::vec3& up, float fovx = 45.f, unsigned int _imageW = 640, unsigned int _imageH = 480)
    {
        Setup(_pos, target, up, fovx, _imageW, _imageH);
    }

    __host__ __device__ void Setup(const glm::vec3& _pos, const glm::vec3& _u, const glm::vec3& _v, const glm::vec3& _w, float fovx, unsigned int _imageW, unsigned int _imageH)
    {
        pos = _pos;
        u = _u;
        v = _v;
        w = _w;
        imageW = _imageW;
        imageH = _imageH;
        aspectRatio = (float)imageW / (float)imageH;
        tanFovxOverTwo = tanf(fovx * 0.5f * M_PI / 180.f);
    }

    __host__ __device__ void Setup(const glm::vec3& _pos, const glm::vec3& target, const glm::vec3& up, float fovx, unsigned int _imageW, unsigned int _imageH)
    {
        pos = _pos;
        w = normalize(pos - target);
        u = cross(up, w);
        v = cross(w, u);
        imageW = _imageW;
        imageH = _imageH;
        aspectRatio = (float)imageW / (float)imageH;
        tanFovxOverTwo = tanf(fovx * 0.5f * M_PI / 180.f);
    }

    // TODO: depth of field
    __device__ void GenerateRay(unsigned int x, unsigned int y, curandState& rng, cudaRay* ray) const
    {
        float nx = 2.f * ((x + curand_uniform(&rng)) / (imageW - 1.f)) - 1.f;
        float ny = 2.f * ((y + curand_uniform(&rng)) / (imageH - 1.f)) - 1.f;

        nx = nx * aspectRatio * tanFovxOverTwo;
        ny = ny * tanFovxOverTwo;

        ray->orig = pos;
        ray->dir = normalize(nx * u + ny * v - w);
    }

public:
    unsigned int imageW, imageH;
    float aspectRatio;
    float tanFovxOverTwo;
    glm::vec3 pos;
    glm::vec3 u, v, w;
};

#endif //SUNPATHTRACER_CAMERA_H

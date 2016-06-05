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

#ifdef __CUDACC__
#include "sampling.h"
#endif

class cudaCamera
{
public:
    __host__ __device__ cudaCamera() {}

    __host__ __device__ cudaCamera(const glm::vec3& _pos, const glm::vec3& _u, const glm::vec3& _v, const glm::vec3& _w, float fovx = 45.f, float apeture = 0.f, float focalLength = 0.f, float exposure = 1.f, unsigned int _imageW = 640, unsigned int _imageH = 480)
    {
        Setup(_pos, _u, _v, _w, fovx, apeture, focalLength, exposure, _imageW, _imageH);
    }

    __host__ __device__ cudaCamera(const glm::vec3& _pos, const glm::vec3& target, const glm::vec3& up, float fovx = 45.f, float apeture = 0.f, float focalLength = 0.f, float exposure = 1.f, unsigned int _imageW = 640, unsigned int _imageH = 480)
    {
        Setup(_pos, target, up, fovx, apeture, focalLength, exposure, _imageW, _imageH);
    }

    __host__ __device__ void Setup(const glm::vec3& _pos, const glm::vec3& _u, const glm::vec3& _v, const glm::vec3& _w, float fovx, float _apeture, float _focalLength, float _exposure, unsigned int _imageW, unsigned int _imageH)
    {
        pos = _pos;
        u = _u;
        v = _v;
        w = _w;
        imageW = _imageW;
        imageH = _imageH;
        aspectRatio = (float)imageW / (float)imageH;
        tanFovxOverTwo = tanf(fovx * 0.5f * M_PI / 180.f);
        exposure = _exposure;
        focalLength = _focalLength;
        apeture = _apeture;
    }

    __host__ __device__ void Setup(const glm::vec3& _pos, const glm::vec3& target, const glm::vec3& up, float fovx, float _apeture, float _focalLength, float _exposure, unsigned int _imageW, unsigned int _imageH)
    {
        pos = _pos;
        w = normalize(pos - target);
        u = cross(up, w);
        v = cross(w, u);
        imageW = _imageW;
        imageH = _imageH;
        aspectRatio = (float)imageW / (float)imageH;
        tanFovxOverTwo = tanf(fovx * 0.5f * M_PI / 180.f);
        exposure = _exposure;
        focalLength = _focalLength;
        apeture = _apeture;
    }

#ifdef __CUDACC__
    __device__ void GenerateRay(unsigned int x, unsigned int y, curandState& rng, cudaRay* ray) const
    {
        float nx = 2.f * ((x + curand_uniform(&rng)) / (imageW - 1.f)) - 1.f;
        float ny = 2.f * ((y + curand_uniform(&rng)) / (imageH - 1.f)) - 1.f;

        nx = nx * aspectRatio * tanFovxOverTwo;
        ny = ny * tanFovxOverTwo;

        //ray->orig = pos;
        //ray->dir = normalize(nx * u + ny * v - w);

        nx = nx * focalLength;
        ny = ny * focalLength;

        glm::vec2 apetureSample = uniform_sample_disk(rng, apeture);
        ray->orig = pos + apetureSample.x * u + apetureSample.y * v;
        ray->dir = glm::normalize((nx - apetureSample.x) * u + (ny - apetureSample.y) * v - focalLength * w);
    }
#endif

public:
    unsigned int imageW, imageH;
    float exposure;
    float apeture;
    float focalLength;
    float aspectRatio;
    float tanFovxOverTwo;
    glm::vec3 pos;
    glm::vec3 u, v, w;
};

#endif //SUNPATHTRACER_CAMERA_H

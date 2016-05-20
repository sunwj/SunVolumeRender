//
// Created by 孙万捷 on 16/2/27.
//

#ifndef SUNPATHTRACER_ONB_H
#define SUNPATHTRACER_ONB_H

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include <cuda_runtime.h>

class cudaONB
{
public:
    __device__ cudaONB(const glm::vec3& _w)
    {
        InitFromW(_w);
    }

    __device__ cudaONB(const glm::vec3& _v, const glm::vec3& _w)
    {
        InitFromVW(_v, _w);
    }

    __device__ void InitFromW(const glm::vec3& _w)
    {
        w = _w;
        if(fabsf(w.x) > fabsf(w.y))
        {
            float invLength = rsqrtf(w.x * w.x + w.z * w.z);
            v = glm::vec3(-w.z * invLength, 0.f, w.x * invLength);
        }
        else
        {
            float  invLength = rsqrtf(w.y * w.y + w.z * w.z);
            v = glm::vec3(0.f, w.z * invLength, -w.y * invLength);
        }
        u = cross(v, w);
    }

    __device__ void InitFromVW(const glm::vec3& _v, const glm::vec3& _w)
    {
        w = _w;
        u = cross(_v, w);
        v = cross(w, u);
    }
public:
    glm::vec3 u, v, w;
};

#endif //SUNPATHTRACER_ONB_H

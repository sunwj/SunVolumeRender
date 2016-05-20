//
// Created by 孙万捷 on 16/5/19.
//

#ifndef SUNVOLUMERENDER_VOLUME_H
#define SUNVOLUMERENDER_VOLUME_H

#include <cuda_runtime.h>

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include "core/cuda_bbox.h"

class cudaVolume
{
public:
    __device__ float operator ()(const glm::vec3& pointInWorld)
    {
        return GetIntensity(pointInWorld);
    }

    __device__ float operator ()(const glm::vec3& normalizedTexCoord, bool dummy)
    {
        return GetIntensityNTC(normalizedTexCoord);
    }

    __device__ bool Intersect(const cudaRay& ray, float* tNear, float* tFar)
    {
        return bbox.Intersect(ray, tNear, tFar);
    }

    __device__ glm::vec3 Gradient_CentralDiff(const glm::vec3& pointInWorld) const
    {
        auto xdiff = GetIntensity(pointInWorld + glm::vec3(spacing.x, 0.f, 0.f)) - GetIntensity(pointInWorld - glm::vec3(spacing.x, 0.f, 0.f));
        auto ydiff = GetIntensity(pointInWorld + glm::vec3(0.f, spacing.y, 0.f)) - GetIntensity(pointInWorld - glm::vec3(0.f, spacing.y, 0.f));
        auto zdiff = GetIntensity(pointInWorld + glm::vec3(0.f, 0.f, spacing.z)) - GetIntensity(pointInWorld - glm::vec3(0.f, 0.f, spacing.z));

        return glm::vec3(xdiff, ydiff, zdiff) * 0.5f;
    }

    __device__ glm::vec3 NormalizedGradient(const glm::vec3& pointInWorld) const
    {
        return glm::normalize(Gradient_CentralDiff(pointInWorld));
    }

private:
    __device__ glm::vec3 GetNormalizedTexCoord(const glm::vec3& pointInWorld) const
    {
        return (pointInWorld - bbox.vmin) * bbox.invSize;
    }

    __device__ float GetIntensity(const glm::vec3& pointInWorld) const
    {
#ifdef __CUDACC__
        auto texCoord = GetNormalizedTexCoord(pointInWorld);
        return tex3D<float>(tex, texCoord.x, texCoord.y, texCoord.z);
#else
        return 0.f;
#endif
    }

    __device__ float GetIntensityNTC(const glm::vec3& normalizedTexCoord)
    {
#ifdef __CUDACC__
        return tex3D<float>(tex, normalizedTexCoord.x, normalizedTexCoord.y, normalizedTexCoord.z);
#else
        return 0.f;
#endif
    }

private:
    cudaBBox bbox;
    cudaTextureObject_t tex = 0;
    glm::vec3 spacing = glm::vec3(glm::uninitialize);
    glm::vec3 invSpacing = glm::vec3(glm::uninitialize);
};

#endif //SUNVOLUMERENDER_VOLUME_H

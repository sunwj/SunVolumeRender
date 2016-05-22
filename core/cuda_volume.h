//
// Created by 孙万捷 on 16/5/19.
//

#ifndef SUNVOLUMERENDER_VOLUME_H
#define SUNVOLUMERENDER_VOLUME_H

#include <cuda_runtime.h>

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include "geometry/cuda_bbox.h"

class cudaVolume
{
public:
    __host__ __device__ void Set(const cudaBBox& bbox, const glm::vec3& spacing, const cudaTextureObject_t& tex)
    {
        this->bbox = bbox;
        this->spacing = spacing;
        this->invSpacing = 1.f / spacing;
        this->tex = tex;
    }

    __device__ float operator ()(const glm::vec3& pointInWorld) const
    {
        return GetIntensity(pointInWorld);
    }

    __device__ float operator ()(const glm::vec3& normalizedTexCoord, bool dummy) const
    {
        return GetIntensityNTC(normalizedTexCoord);
    }

    __device__ bool Intersect(const cudaRay& ray, float* tNear, float* tFar) const
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

    __device__ bool IsInside(const glm::vec3& ptInWorld) const
    {
        return bbox.IsInside(ptInWorld);
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

    __device__ float GetIntensityNTC(const glm::vec3& normalizedTexCoord) const
    {
#ifdef __CUDACC__
        return tex3D<float>(tex, normalizedTexCoord.x, normalizedTexCoord.y, normalizedTexCoord.z);
#else
        return 0.f;
#endif
    }

private:
    cudaBBox bbox;
    cudaTextureObject_t tex;
    glm::vec3 spacing;
    glm::vec3 invSpacing;
};

#endif //SUNVOLUMERENDER_VOLUME_H

//
// Created by 孙万捷 on 16/5/4.
//

#ifndef SUNPATHTRACER_CUDA_ENVIRONMENT_LIGHT_H
#define SUNPATHTRACER_CUDA_ENVIRONMENT_LIGHT_H

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include <cuda_runtime.h>

class cudaEnvironmentLight
{
public:
    __host__ __device__ cudaEnvironmentLight() {}

    __host__ __device__ void Set(cudaTextureObject_t tex)
    {
        this->tex = tex;
        this->intensity = 1.f;
        this->offset = glm::vec2(0.f);
    }

    __host__ __device__ void Set(const glm::vec3& radiance)
    {
        tex = 0;
        defaultRadiance = radiance;
        this->intensity = 1.f;
        this->offset = glm::vec2(0.f);
    }

    __host__ __device__ void SetIntensity(float intensity)
    {
        this->intensity = intensity;
    }

    __host__ __device__ void SetOffset(const glm::vec2& offset)
    {
        this->offset = offset;
    }

    __host__ __device__ cudaTextureObject_t Get()
    {
        return tex;
    }

    __device__ glm::vec3 GetEnvRadiance(const glm::vec2& texcoord)
    {
#ifdef __CUDACC__
        auto val = tex2D<float4>(tex, texcoord.x, texcoord.y);
        return tex ? glm::vec3(val.x, val.y, val.z) * intensity : defaultRadiance * intensity;
#else
        return glm::vec3(0.f);
#endif
    }

    __device__ glm::vec3 GetEnvRadiance(const glm::vec3& dir)
    {
        float theta = acosf(dir.y);
        float phi = atan2f(dir.x, dir.z);
        phi = phi < 0.f ? phi + 2.f * M_PI : phi;
        float u = phi * 0.5f * M_1_PI;
        float v = theta * M_1_PI;

#ifdef __CUDACC__
        auto val = tex2D<float4>(tex, u + offset.x, v + offset.y);
        return tex ? glm::vec3(val.x, val.y, val.z) * intensity : defaultRadiance * intensity;
#else
        return glm::vec3(0.f);
#endif
    }

private:
    cudaTextureObject_t tex;
    glm::vec3 defaultRadiance;
    float intensity;
    glm::vec2 offset;
};

#endif //SUNPATHTRACER_CUDA_ENVIRONMENT_LIGHT_H

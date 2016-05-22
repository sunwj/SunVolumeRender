//
// Created by 孙万捷 on 16/5/20.
//

#ifndef SUNVOLUMERENDER_CUDA_TRANSFER_FUNCTION_H
#define SUNVOLUMERENDER_CUDA_TRANSFER_FUNCTION_H

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include <cuda_runtime.h>

class cudaTransferFunction
{
public:
    __host__ __device__ void Set(const cudaTextureObject_t& tex)
    {
        this->tex = tex;
    }

    __device__ glm::vec4 operator ()(float intensity) const
    {
#ifdef __CUDACC__
        auto val = tex1D<float4>(tex, intensity);
        return glm::vec4(val.x, val.y, val.z, val.w);
#else
        return glm::vec4(0.f);
#endif
    }

    __device__ glm::vec3 GetColor(float intensity) const
    {
#ifdef __CUDACC__
        auto val = tex1D<float4>(tex, intensity);
        return glm::vec3(val.x, val.y, val.z);
#else
        return glm::vec3(0.f);
#endif
    }

    __device__ float GetOpacity(float intensity) const
    {
#ifdef __CUDACC__
        auto val = tex1D<float4>(tex, intensity);
        return val.w;
#else
        return 0.f;
#endif
    }

private:
    cudaTextureObject_t tex;
};

#endif //SUNVOLUMERENDER_CUDA_TRANSFER_FUNCTION_H

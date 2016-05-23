//
// Created by 孙万捷 on 16/5/22.
//

#ifndef SUNVOLUMERENDER_HENYEY_GREENSTEIN_H
#define SUNVOLUMERENDER_HENYEY_GREENSTEIN_H

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include <curand_kernel.h>

#include "../cuda_onb.h"

__inline__ __device__ float hg_phase_f(const glm::vec3& wo, const glm::vec3& wi, float g = 0.f)
{
    if(g == 0)
        return M_1_PI * 0.25f;

    auto val = (1.f - g * g) / powf(1.f + g * g - 2.f * g * glm::dot(wi, wo), 1.5);
    return val * M_1_PI * 0.25f;
}

__inline__ __device__ float hg_phase_pdf(const glm::vec3& wo, const glm::vec3& wi, float g = 0)
{
    return hg_phase_f(wo, wi, g);
}

__inline__ __device__ void hg_phase_sample_f(float g, const glm::vec3& wo, glm::vec3* wi, float* pdf, curandState& rng)
{
    float cosTheta = 0.f;
    float phi = 2.f * M_PI * curand_uniform(&rng);

    if(g == 0)
    {
        cosTheta = 1.f - 2.f * curand_uniform(&rng);
    }
    else
    {
        float tmp = (1.f - g * g) / (1.f - g + 2.f * g * curand_uniform(&rng));
        cosTheta = (1.f + g * g - tmp * tmp) / (2.f * g);
    }

    float sinTheta = sqrtf(fmaxf(0.f, 1.f - cosTheta * cosTheta));

    cudaONB onb(wo);
    *wi = glm::normalize(sinTheta * cosf(phi) * onb.u + sinTheta * sinf(phi) * onb.v + cosTheta * onb.w);

    if(pdf)
        *pdf = hg_phase_pdf(wo, *wi, g);
}

#endif //SUNVOLUMERENDER_HENYEY_GREENSTEIN_H

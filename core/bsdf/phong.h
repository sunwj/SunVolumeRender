//
// Created by 孙万捷 on 16/5/24.
//

#ifndef SUNVOLUMERENDER_PHONG_H
#define SUNVOLUMERENDER_PHONG_H

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "../sampling.h"

__inline__ __device__ float phong_brdf_f(const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& normal, float e)
{
    auto r = glm::reflect(-wo, normal);
    float cosTerm = fmaxf(0.f, glm::dot(r, wi));

    return powf(cosTerm, e);
}

__inline__ __device__ float phong_brdf_pdf(const glm::vec3& r, const glm::vec3& wi, float e)
{
    float cosTerm = glm::dot(r, wi);
    if(cosTerm <= 0.f)
        return 0.f;

    return (e + 1.f) * 0.5f * M_1_PI * powf(cosTerm, e);
}

__inline__ __device__ void phong_brdf_sample_f(float e, const glm::vec3& normal, const glm::vec3& wo, glm::vec3* wi, float* pdf, curandState& rng)
{
    auto r = glm::reflect(-wo, normal);
    *wi = sample_phong(rng, e, r);

    if(pdf)
        *pdf = phong_brdf_pdf(r, *wi, e);
}

#endif //SUNVOLUMERENDER_PHONG_H

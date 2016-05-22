//
// Created by 孙万捷 on 16/5/22.
//

#ifndef SUNVOLUMERENDER_HENYEY_GREENSTEIN_H
#define SUNVOLUMERENDER_HENYEY_GREENSTEIN_H

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

__inline__ __device__ float hg_phase_evaluate(const glm::vec3& wi, const glm::vec3& wo, float g)
{
    auto val = (1.f - g * g) / powf(1.f + g * g - 2.f * g * glm::dot(wi, wo), 1.5);
    return val * M_1_PI * 0.25f;
}

#endif //SUNVOLUMERENDER_HENYEY_GREENSTEIN_H

//
// Created by 孙万捷 on 16/5/22.
//

#ifndef SUNVOLUMERENDER_TRANSMITTANCE_H
#define SUNVOLUMERENDER_TRANSMITTANCE_H

#include "woodcock_tracking.h"

__inline__ __device__ float transmittance(const glm::vec3& start, const glm::vec3& end, const cudaVolume& volume, const cudaTransferFunction& tf, curandState& rng)
{
    cudaRay ray(start, glm::normalize(end - start));

    auto t = sample_distance(ray, volume, tf, rng);
    auto flag = (t > ray.tMin) && (t < ray.tMax);
    return flag ? 0.f : 1.f;
}

#endif //SUNVOLUMERENDER_TRANSMITTANCE_H

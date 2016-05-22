//
// Created by 孙万捷 on 16/5/22.
//

#ifndef SUNVOLUMERENDER_TRANSMITTANCE_H
#define SUNVOLUMERENDER_TRANSMITTANCE_H

#include "woodcock_tracking.h"

__inline__ __device__ float transmittance(const cudaRay& ray, const cudaVolume& volume, const cudaTransferFunction& tf, float invSigmaMax, curandState& rng)
{
    auto t = sample_distance(ray, volume, tf, invSigmaMax, rng);
    auto flag = (t > ray.tMin) && (t < ray.tMax);
    return flag ? 0.f : 1.f;
}

#endif //SUNVOLUMERENDER_TRANSMITTANCE_H

//
// Created by 孙万捷 on 16/5/25.
//

#ifndef SUNVOLUMERENDER_FRESNEL_H
#define SUNVOLUMERENDER_FRESNEL_H

#include <cuda_runtime.h>

__inline__ __device__ float schlick_fresnel(float ni, float no, float cosin)
{
    float R0 = (ni - no) * (ni - no) / ((ni + no) * (ni + no));
    float c = 1.f - cosin;
    return R0 + (1.f - R0) * c * c * c * c * c;
}

#endif //SUNVOLUMERENDER_FRESNEL_H

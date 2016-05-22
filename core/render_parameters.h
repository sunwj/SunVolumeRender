//
// Created by 孙万捷 on 16/5/22.
//

#ifndef SUNVOLUMERENDER_RENDER_PARAMETERS_H
#define SUNVOLUMERENDER_RENDER_PARAMETERS_H

#include <cuda_runtime.h>

#include "../utils/helper_cuda.h"

class RenderParams
{
public:
    __host__ RenderParams() {}

    __host__ void SetupHDRBuffer(uint32_t w, uint32_t h)
    {
        Clear();

        checkCudaErrors(cudaMalloc(&hdrBuffer, sizeof(glm::vec3) * w * h));
        checkCudaErrors(cudaMemset(hdrBuffer, 0, sizeof(glm::vec3) * w * h));
    }

    __host__ void Clear()
    {
        if(hdrBuffer)
        {
            checkCudaErrors(cudaFree(hdrBuffer));
            hdrBuffer = nullptr;
        }
    }

public:
    uint32_t traceDepth = 1;
    uint32_t frameNo = 0;
    float exposure = 1.f;
    float maxOpacity = 0.5f;
    glm::vec3* hdrBuffer = nullptr;
};

#endif //SUNVOLUMERENDER_RENDER_PARAMETERS_H

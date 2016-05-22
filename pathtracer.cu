//
// Created by 孙万捷 on 16/3/4.
//

#define GLM_FORCE_NO_CTOR_INIT
#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "utils/helper_cuda.h"
#include "core/cuda_camera.h"
#include "core/cuda_transfer_function.h"
#include "core/cuda_volume.h"
#include "core/render_parameters.h"
#include "core/tonemapping.h"
#include "core/woodcock_tracking.h"
#include "core/transmittance.h"
#include "core/bsdf/henyey_greenstein.h"
#include "common.h"

// global variables
__constant__ cudaTransferFunction transferFunction;
extern "C" void setup_transferfunction(const cudaTransferFunction& tf)
{
    checkCudaErrors(cudaMemcpyToSymbol(transferFunction, &tf, sizeof(cudaTransferFunction), 0));
    checkCudaErrors(cudaDeviceSynchronize());
}

__constant__ cudaVolume volume;
extern "C" void setup_volume(const cudaVolume& vol)
{
    checkCudaErrors(cudaMemcpyToSymbol(volume, &vol, sizeof(cudaVolume), 0));
    checkCudaErrors(cudaDeviceSynchronize());
}

__constant__ cudaCamera camera;
extern "C" void setup_camera(const cudaCamera& cam)
{
    checkCudaErrors(cudaMemcpyToSymbol(camera, &cam, sizeof(cudaCamera), 0));
    checkCudaErrors(cudaDeviceSynchronize());
}

__host__ __device__ uint32_t wangHash(uint32_t a)
{
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);

    return a;
}

__inline__ __device__ void running_estimate(glm::vec3& acc_buffer, const glm::vec3& curr_est, unsigned int N)
{
    acc_buffer += (curr_est - acc_buffer) / (N + 1.f);
}

template <typename T>
__global__ void clear_hdr_buffer(T* buffer)
{
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    auto idy = blockDim.y * blockIdx.y + threadIdx.y;
    auto offset = idy * WIDTH + idx;

    buffer[offset] = T(0.f);
}

__global__ void render_kernel(const RenderParams renderParams, uint32_t hashedFrameNo)
{
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    auto idy = blockDim.y * blockIdx.y + threadIdx.y;
    auto offset = idy * WIDTH + idx;
    curandState rng;
    curand_init(hashedFrameNo + offset, 0, 0, &rng);

    glm::vec3 L = glm::vec3(0.f);
    glm::vec3 T = glm::vec3(1.f);

    cudaRay ray;
    camera.GenerateRay(idx, idy, rng, &ray);

    float tNear, tFar;
    if(!volume.Intersect(ray, &tNear, &tFar))
    {
        running_estimate(renderParams.hdrBuffer[offset], L, renderParams.frameNo);
        return;
    }

    ray.tMin = tNear;
    ray.tMax = tFar;
    auto invSigmaMax = opacity_to_sigmat(renderParams.maxOpacity);
    for(auto k = 0; k < renderParams.traceDepth; ++k)
    {
        auto t = sample_distance(ray, volume, transferFunction, invSigmaMax, rng);
        if(t < 0.f)
        {
            break;
        }

        auto ptInWorld = ray.PointOnRay(t);
        auto intensity = volume(ptInWorld);
        auto color_opacity = transferFunction(intensity);
        auto albedo = glm::vec3(color_opacity.x, color_opacity.y, color_opacity.z);
        auto opacity = color_opacity.w;

        cudaRay shadowRay(ptInWorld, glm::normalize(glm::vec3(1.f)));
        volume.Intersect(shadowRay, &tNear, &tFar);
        shadowRay.tMax = tFar;
        auto transmittion = transmittance(shadowRay, volume, transferFunction, invSigmaMax, rng);

        T *= albedo;
        L += transmittion * T * hg_phase_evaluate(shadowRay.dir, -ray.dir, 0.f) * 5.f;
    }

    running_estimate(renderParams.hdrBuffer[offset], L, renderParams.frameNo);
}

__global__ void hdr_to_ldr(glm::u8vec4* img, const RenderParams renderParams)
{
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    auto idy = blockDim.y * blockIdx.y + threadIdx.y;
    auto offset = idy * WIDTH + idx;

    auto L = reinhard_tone_mapping(renderParams.hdrBuffer[offset], renderParams.exposure);
    img[offset] = glm::u8vec4(L.x * 255, L.y * 255, L.z * 255, 255);
}

extern "C" void rendering(glm::u8vec4* img, const RenderParams& renderParams)
{
    dim3 blockSize(16, 16);
    dim3 gridSize(WIDTH / blockSize.x, HEIGHT / blockSize.y);

    if(renderParams.frameNo == 0)
    {
        clear_hdr_buffer<<<gridSize, blockSize>>>(renderParams.hdrBuffer);
    }

    render_kernel<<<gridSize, blockSize>>>(renderParams, wangHash(renderParams.frameNo));
    hdr_to_ldr<<<gridSize, blockSize>>>(img, renderParams);
}
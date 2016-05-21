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
#include "common.h"

// global variables
__constant__ cudaTransferFunction transferFunction;
extern "C" void setup_transferfunction(const cudaTransferFunction& tf)
{
    checkCudaErrors(cudaMemcpyToSymbol(transferFunction, &tf, sizeof(cudaTransferFunction), 0));
}

__constant__ cudaVolume volume;
extern "C" void setup_volume(const cudaVolume& vol)
{
    checkCudaErrors(cudaMemcpyToSymbol(volume, &vol, sizeof(cudaVolume), 0));
}

__host__ __device__ unsigned int wangHash(unsigned int a)
{
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);

    return a;
}

template <typename T>
__global__ void clear_hdr_buffer(T* buffer)
{
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    auto idy = blockDim.y * blockIdx.y + threadIdx.y;
    auto offset = idy * WIDTH + idx;

    buffer[offset] = T(0.f);
}

__global__ void render_kernel(glm::u8vec4* img, const cudaCamera camera, unsigned int hashedFrameNo)
{
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    auto idy = blockDim.y * blockIdx.y + threadIdx.y;
    auto offset = idy * WIDTH + idx;
    curandState rng;
    curand_init(hashedFrameNo + offset, 0, 0, &rng);

    cudaRay ray;
    camera.GenerateRay(idx, idy, rng, &ray);

    float tNear, tFar;
    if(!volume.Intersect(ray, &tNear, &tFar))
    {
        img[offset] = glm::u8vec4(0, 0, 0, 0);
        return;
    }

    auto val = volume(ray.PointOnRay(tNear));
    img[offset] = glm::u8vec4(val * 255, val * 255, val * 255, 255);
}

extern "C" void rendering(glm::u8vec4* img, const cudaCamera& camera, unsigned int frameNo)
{
    dim3 blockSize(16, 16);
    dim3 gridSize(WIDTH / blockSize.x, HEIGHT / blockSize.y);

    render_kernel<<<gridSize, blockSize>>>(img, camera, frameNo);
}
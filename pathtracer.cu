//
// Created by 孙万捷 on 16/3/4.
//

#define GLM_FORCE_NO_CTOR_INIT
#include <stdio.h>

#include <glm/glm.hpp>

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
#include "core/bsdf/henyey_greenstein.h"
#include "core/bsdf/phong.h"
#include "core/bsdf/fresnel.h"
#include "core/lights/cuda_lights.h"
#include "core/lights/lights.h"
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

__constant__ cudaLights lights;
extern "C" void setup_lights(const Lights& hostLights)
{
    checkCudaErrors(cudaMemcpyToSymbol(lights.environmentLight, &(hostLights.environmentLight), sizeof(cudaEnvironmentLight), 0));
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

__inline__ __device__ bool terminate_with_raussian_roulette(glm::vec3* troughput, curandState& rng)
{
    float illum = 0.2126f * troughput->x + 0.7152f * troughput->y + 0.0722 * troughput->z;
    if(curand_uniform(&rng) > illum) return true;
    *troughput /= illum;

    return false;
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

    auto invSigmaMax = 1.f / opacity_to_sigmat(renderParams.maxOpacity);
    for(auto k = 0; k < renderParams.traceDepth; ++k)
    {
        auto t = sample_distance(ray, volume, transferFunction, invSigmaMax, rng);
        if(t < 0.f)
        {
            L += T * lights.GetEnvironmentRadiance(ray.dir, renderParams.envLightOffset.x, renderParams.envLightOffset.y);
            break;
        }

        auto ptInWorld = ray.PointOnRay(t);
        auto intensity = volume(ptInWorld);
        auto gradient = volume.Gradient_CentralDiff(ptInWorld);
        auto gradientMagnitude = sqrtf(glm::dot(gradient, gradient));
        auto color_opacity = transferFunction(intensity);
        auto albedo = glm::vec3(color_opacity.x, color_opacity.y, color_opacity.z);
        auto opacity = color_opacity.w;

        auto wi = glm::normalize(glm::vec3(1.f, 0.f, -1.f));
        //auto Tl = transmittance(ptInWorld, ptInWorld + wi, volume, transferFunction, invSigmaMax, rng);
        auto Tl = 0.f;

        if(gradientMagnitude < 1e-3)
        {
            L += T * Tl * albedo * hg_phase_f(-ray.dir, wi, 0.f);

            glm::vec3 newDir;
            hg_phase_sample_f(0.f, -ray.dir, &newDir, nullptr, rng);
            ray.orig = ptInWorld;
            ray.dir = newDir;

            T *= albedo;
        }
        else
        {
            auto normal = glm::normalize(gradient);
            if(glm::dot(normal, ray.dir) > 0.f)
                normal = -normal;

            float ks = schlick_fresnel(1.0f, 2.5f, glm::dot(-ray.dir, normal));
            float kd = 1.f - ks;

            auto diffuse = albedo * (float)M_1_PI * 0.5f;
            auto specular = glm::vec3(1.f) * phong_brdf_f(-ray.dir, wi, normal, 20.f);

            auto cosTerm = fmaxf(0.f, glm::dot(normal, wi));
            L += T * Tl * (kd * diffuse + ks * specular) * cosTerm;

            auto p = 0.25f + 0.5f * ks;
            if(curand_uniform(&rng) < p)
            {
                ray.orig = ptInWorld;
                ray.dir = sample_phong(rng, 20.f, glm::reflect(ray.dir, normal));

                T *= ks / p;
            }
            else
            {
                ray.orig = ptInWorld;
                ray.dir = cosine_weightd_sample_hemisphere(rng, normal);

                T *= albedo;
                T *= kd / (1.f - p);
            }
        }

        if(k >= 3)
        {
            if(terminate_with_raussian_roulette(&T, rng))
                break;
        }
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
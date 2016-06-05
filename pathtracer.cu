//
// Created by 孙万捷 on 16/3/4.
//

#define GLM_FORCE_NO_CTOR_INIT
#include <stdio.h>

#define GLM_FORCE_INLINE
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
#include "core/transmittance.h"
#include "core/bsdf/henyey_greenstein.h"
#include "core/bsdf/phong.h"
#include "core/bsdf/fresnel.h"
#include "core/lights/lights.h"
#include "core/lights/cuda_environment_light.h"
#include "core/lights/light_sample.h"
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

__constant__ uint32_t num_areaLights;
__constant__ cudaAreaLight areaLights[MAX_LIGHT_SOURCES];
extern "C" void setup_area_lights(cudaAreaLight* lights, uint32_t n)
{
    checkCudaErrors(cudaMemcpyToSymbol(num_areaLights, &n, sizeof(uint32_t), 0));
    checkCudaErrors(cudaMemcpyToSymbol(areaLights, lights, sizeof(cudaAreaLight) * n, 0));
}

__constant__ cudaEnvironmentLight envLight;
extern "C" void setup_env_lights(const cudaEnvironmentLight& light)
{
    checkCudaErrors(cudaMemcpyToSymbol(envLight, &light, sizeof(cudaEnvironmentLight), 0));
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

enum ShadingType{SHANDING_TYPE_ISOTROPIC, SHANDING_TYPE_BRDF};
__inline__ __device__ glm::vec3 bsdf(const VolumeSample vs, const glm::vec3& wi, ShadingType st)
{
    glm::vec3 diffuseColor = glm::vec3(vs.color_opacity.x, vs.color_opacity.y, vs.color_opacity.z);

    glm::vec3 L;
    if(st == SHANDING_TYPE_ISOTROPIC)
    {
         L = diffuseColor * hg_phase_f(vs.wo, wi);
    }
    else if(st == SHANDING_TYPE_BRDF)
    {
        auto normal = glm::normalize(vs.gradient);

        float cosTerm = fabsf(glm::dot(wi, normal));
        float ks = schlick_fresnel(1.0f, 2.5f, cosTerm);
        float kd = 1.f - ks;

        auto diffuse = diffuseColor * (float)M_1_PI * 0.5f;
        auto specular = glm::vec3(1.f) * phong_brdf_f(vs.wo, wi, normal, 20.f);

        L = (kd * diffuse + ks * specular) * cosTerm;
    }

    return L;
}

__inline__ __device__ glm::vec3 estimate_direct_light(const VolumeSample vs, curandState& rng, ShadingType st)
{
    glm::vec3 Li = glm::vec3(0.f);

    if(num_areaLights == 0)
        return Li;

    // randomly choose a single light
    int lightId = num_areaLights * curand_uniform(&rng);
    lightId = lightId < num_areaLights ? lightId : num_areaLights - 1;
    const cudaAreaLight& light = areaLights[lightId];

    // sample light
    glm::vec3 lightPos;
    glm::vec3 wi;
    float pdf;
    Li = sample_light(light, vs.ptInWorld, rng, &lightPos, &wi, &pdf);

    if(pdf > 0.f && fmaxf(Li.x, fmaxf(Li.y, Li.z)) > 0.f)
    {
        auto Tr = transmittance(vs.ptInWorld, lightPos, volume, transferFunction, rng);
        Li = Tr * num_areaLights * bsdf(vs, wi, st) * Li / pdf;
    }

    return Li;
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

    LightSample ls;
    bool hitLight = get_nearest_light_sample(ray, areaLights, num_areaLights, &ls);
    for(auto k = 0; k < renderParams.traceDepth; ++k)
    {
        auto t = sample_distance(ray, volume, transferFunction, rng);

        if((k == 0) && hitLight)
        {
            t = t < 0.f ? FLT_MAX : t;
            if(ls.t < t)
            {
                auto cosTerm = glm::dot(ls.normal, -ray.dir);
                L += T * ls.radiance * 1000.f * (cosTerm <= 0.f ? 0.f : 1.f);
                break;
            }
        }

        if(t < 0.f)
        {
            L += T * envLight.GetEnvRadiance(ray.dir);
            break;
        }

        VolumeSample vs;

        vs.wo = -ray.dir;
        vs.ptInWorld = ray.PointOnRay(t);
        vs.intensity = volume(vs.ptInWorld);
        vs.color_opacity = transferFunction(vs.intensity);
        vs.gradient = volume.Gradient_CentralDiff(vs.ptInWorld);
        vs.gradientMagnitude = sqrtf(glm::dot(vs.gradient, vs.gradient));

        if(vs.gradientMagnitude < 1e-3)
        {
            L += T * estimate_direct_light(vs, rng, SHANDING_TYPE_ISOTROPIC);
            //L += T * glm::vec3(0.1f);

            glm::vec3 newDir;
            hg_phase_sample_f(0.f, -ray.dir, &newDir, nullptr, rng);
            ray.orig = vs.ptInWorld;
            ray.dir = newDir;

            T *= glm::vec3(vs.color_opacity.x, vs.color_opacity.y, vs.color_opacity.z);
        }
        else
        {
            auto normal = glm::normalize(vs.gradient);
            if(glm::dot(normal, vs.wo) < 0.f)
                normal = -normal;

            L += T * estimate_direct_light(vs, rng, SHANDING_TYPE_BRDF);
            //L += T * glm::vec3(0.1f);

            auto cosTerm = fmaxf(0.f, glm::dot(vs.wo, normal));
            auto ks = schlick_fresnel(1.f, 2.5f, cosTerm);
            auto kd = 1.f - ks;
            auto p = 0.25f + 0.5f * ks;
            if(curand_uniform(&rng) < p)
            {
                ray.orig = vs.ptInWorld;
                ray.dir = sample_phong(rng, 20.f, glm::reflect(ray.dir, normal));

                T *= ks / p;
            }
            else
            {
                ray.orig = vs.ptInWorld;
                ray.dir = cosine_weightd_sample_hemisphere(rng, normal);

                T *= glm::vec3(vs.color_opacity.x, vs.color_opacity.y, vs.color_opacity.z);
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

    auto L = reinhard_tone_mapping(renderParams.hdrBuffer[offset], camera.exposure);
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
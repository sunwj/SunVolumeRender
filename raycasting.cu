#define GLM_FORCE_NO_CTOR_INIT
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "common.h"
#include "utils/helper_cuda.h"
#include "core/cuda_camera.h"
#include "core/cuda_transfer_function.h"
#include "core/cuda_volume.h"

__global__ void kernel_raycasting(glm::u8vec4* img, cudaVolume volume, cudaTransferFunction transferFunction, cudaCamera camera, float stepSize)
{
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    auto idy = blockDim.y * blockIdx.y + threadIdx.y;
    auto offset = idy * WIDTH + idx;

    cudaRay ray;
    camera.GenerateRay(idx, idy, &ray);

    glm::vec4 L = glm::vec4(0.f);

    float tNear, tFar, t;
    if(volume.Intersect(ray, &tNear, &tFar))
    {
        t = tNear;
        while(t <= tFar)
        {
            auto ptInWorld = ray.PointOnRay(t);
            auto intensity = volume(ptInWorld);
            auto color_opacity = transferFunction(intensity);

            // apply lighting
            auto gradient = volume.Gradient_CentralDiff(ptInWorld);
            auto gradientMagnitude = sqrtf(glm::dot(gradient, gradient));
            float cosTerm = 1.f;
            float specularTerm = 0.f;
            if(gradientMagnitude > 1e-3)
            {
                auto normal = glm::normalize(gradient);
                auto lightDir = glm::normalize(camera.pos - ptInWorld);
                cosTerm = fabsf(glm::dot(normal, lightDir));

                specularTerm = powf(cosTerm, 30.f);
            }

            color_opacity.x = color_opacity.x * color_opacity.w * cosTerm * 0.8f + color_opacity.w * specularTerm * 0.2f;
            color_opacity.y = color_opacity.y * color_opacity.w * cosTerm * 0.8f + color_opacity.w * specularTerm * 0.2f;
            color_opacity.z = color_opacity.z * color_opacity.w * cosTerm * 0.8f + color_opacity.w * specularTerm * 0.2f;

            L += (1.f - L.w) * color_opacity;

            if(L.w > 0.95f) break;

            t += stepSize * 0.5f;
        }

    }

    L.x = fminf(L.x, 1.f);
    L.y = fminf(L.y, 1.f);
    L.z = fminf(L.z, 1.f);
    img[offset] = glm::u8vec4(L.x * 255, L.y * 255, L.z * 255, 255 * L.w);
}

extern "C" void render_raycasting(glm::u8vec4* img, cudaVolume& volume, cudaTransferFunction& transferFunction, cudaCamera& camera, float stepSize)
{
    dim3 blockSize(16, 16);
    dim3 gridSize(WIDTH / blockSize.x, HEIGHT / blockSize.y);

    kernel_raycasting<<<gridSize, blockSize>>>(img, volume, transferFunction, camera, stepSize);
}
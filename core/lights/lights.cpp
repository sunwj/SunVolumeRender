//
// Created by 孙万捷 on 16/5/25.
//

#include "lights.h"
#include "../../utils/helper_cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../utils/stb_image.h"

Lights::Lights()
{
    environmentLight.Set(glm::vec3(0.03f));
}

Lights::~Lights()
{
    if(!environmentLight.Get())
    {
        checkCudaErrors(cudaDestroyTextureObject(environmentLight.Get()));
        environmentLight.Set(0);
    }

    if(envMapArray)
    {
        checkCudaErrors(cudaFreeArray(envMapArray));
        envMapArray = nullptr;
    }
}

void Lights::SetEnvironmentLight(std::string filename)
{
    int w = 0, h = 0, n = 0;
    float* data = stbi_loadf(filename.c_str(), &w, &h, &n, 0);
    if(!data)
    {
        std::cerr<<"Unable to load environment map: "<<filename<<std::endl;
        exit(0);
    }

    //create channel desc
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    //create cudaArray
    checkCudaErrors(cudaMallocArray(&envMapArray, &channelDesc, w, h));
    if(n == 3)
    {
        uint32_t count = w * h;
        std::vector<float4> ext_data;
        ext_data.reserve(count);
        for(auto i = 0; i < count; ++i)
            ext_data.push_back(make_float4(data[i * 3], data[i * 3 + 1], data[i * 3 + 2], 0.f));

        checkCudaErrors(cudaMemcpyToArray(envMapArray, 0, 0, ext_data.data(), sizeof(float4) * w * h, cudaMemcpyHostToDevice));
    }
    else
        checkCudaErrors(cudaMemcpyToArray(envMapArray, 0, 0, data, sizeof(float4) * w * h, cudaMemcpyHostToDevice));
    //create resource desc
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = envMapArray;
    //create texture desc
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = true;
    //create cudaTextureObject
    cudaTextureObject_t tex;
    checkCudaErrors(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    environmentLight.Set(tex);
}

void Lights::SetEnvionmentLight(const glm::vec3 &radiance)
{
    environmentLight.Set(radiance);
}
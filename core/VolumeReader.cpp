//
// Created by 孙万捷 on 16/5/21.
//

#include <driver_types.h>
#include <vtkImageCast.h>
#include <vtkImageGradientMagnitude.h>
#include "VolumeReader.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "utils/stb_image_write.h"

void VolumeReader::Read(std::string filename)
{
    ClearHost();

    auto metaImageReader = vtkSmartPointer<vtkMetaImageReader>::New();

    QFileInfo fileInfo(filename.c_str());
    if(!fileInfo.exists())
    {
        std::cerr<<filename<<" does not exist"<<std::endl;
        exit(0);
    }

    if(!metaImageReader->CanReadFile(filename.c_str()))
    {
        std::cerr<<"meta image reader cannot read "<<filename<<std::endl;
        exit(0);
    }
    metaImageReader->SetFileName(filename.c_str());
    metaImageReader->Update();

    if(metaImageReader->GetErrorCode() != vtkErrorCode::NoError)
    {
        std::cerr<<"Error loading file "<<vtkErrorCode::GetStringFromErrorCode(metaImageReader->GetErrorCode())<<std::endl;
        exit(0);
    }

    auto imageCast = vtkSmartPointer<vtkImageCast>::New();
    imageCast->SetInput(metaImageReader->GetOutput());
    imageCast->SetOutputScalarTypeToShort();
    imageCast->Update();
    auto imageData = imageCast->GetOutput();

    auto dataExtent = imageData->GetExtent();
    this->dim = glm::ivec3(dataExtent[1] + 1, dataExtent[3] + 1, dataExtent[5] + 1);

    auto dataSpacing = imageData->GetSpacing();
    this->spacing = glm::vec3(static_cast<float>(dataSpacing[0]), static_cast<float>(dataSpacing[1]), static_cast<float>(dataSpacing[2]));

    auto noEles = dim.x * dim.y * dim.z;
    volumeData = new char[noEles * 2];
    auto dataRange = imageData->GetScalarRange();
    Rescale<short, unsigned short>(reinterpret_cast<short*>(imageData->GetScalarPointer()), noEles, dataRange[0], dataRange[1]);

    auto hist = vtkSmartPointer<vtkImageAccumulate>::New();
    hist->SetInputConnection(imageCast->GetOutputPort());
    hist->SetComponentExtent(0, dataRange[1] - dataRange[0] - 1, 0, 0, 0, 0);
    hist->SetComponentOrigin(dataRange[0], 0, 0);
    hist->SetComponentSpacing(1, 0, 0);
    hist->IgnoreZeroOn();
    hist->Update();

    auto histDims = hist->GetOutput()->GetDimensions();
    histogram.clear();
    histogram.resize(histDims[0]);
    memcpy(&histogram[0], hist->GetOutput()->GetScalarPointer(), sizeof(uint32_t) * histDims[0]);

    auto imageGradientMagnitude = vtkSmartPointer<vtkImageGradientMagnitude>::New();
    imageGradientMagnitude->SetDimensionality(3);
    imageGradientMagnitude->SetInput(imageCast->GetOutput());
    imageGradientMagnitude->Update();
    auto magData = imageGradientMagnitude->GetOutput();
    auto magRange = magData->GetScalarRange();
    maxMagnitude = magRange[1];
    std::cout<<maxMagnitude<<std::endl;

    //auto ptr = reinterpret_cast<unsigned short*>(data) + 60 * 512 * 512;
    //unsigned char* tmp = new unsigned char[512 * 512 * 3];
    //for(auto i = 0; i < 512; ++i)
    //{
    //    for(auto j = 0; j < 512; ++j)
    //    {
    //        auto offset = i * 512 + j;
    //        tmp[offset * 3] = ptr[offset] / 65535.f * 255;
    //        tmp[offset * 3 + 1] = tmp[offset * 3];
    //        tmp[offset * 3 + 2] = tmp[offset * 3];
    //    }
    //}
//
    //stbi_write_bmp("test.bmp", 512, 512, 3, (char*)tmp);
    //delete []tmp;
}

void VolumeReader::ClearHost()
{
    if(volumeData != nullptr)
    {
        delete volumeData;
        volumeData = nullptr;
    }

    spacing = glm::vec3(0.f);
    dim = glm::ivec3(0);
}

void VolumeReader::ClearDevice()
{
    if(volumeArray != nullptr)
    {
        checkCudaErrors(cudaFreeArray(volumeArray));
        volumeArray = nullptr;
    }

    if(volumeTex != 0)
    {
        checkCudaErrors(cudaDestroyTextureObject(volumeTex));
        volumeTex = 0;
    }
}

template <typename T, typename UT>
void VolumeReader::Rescale(T* dataPtr, size_t size, float dataMin, float dataMax)
{
    auto ptr1 = dataPtr;
    auto ptr2 = reinterpret_cast<UT*>(volumeData);

    auto extent = dataMax - dataMin;
    auto dataTypeExtent = std::numeric_limits<UT>::max() - std::numeric_limits<UT>::min();
    // rescale data to unsigned data type
    for(auto i = 0; i < size; ++i)
    {
        ptr2[i] = (ptr1[i] - dataMin) / extent * dataTypeExtent;
    }
}

void VolumeReader::CreateTextures()
{
    ClearDevice();

    int bytesPerElement = 2;
    // allocate cudaArray
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(bytesPerElement * 8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaExtent extent = make_cudaExtent(dim.x, dim.y, dim.z);
    checkCudaErrors(cudaMalloc3DArray(&volumeArray, &channelDesc, extent, cudaArrayDefault));

    cudaMemcpy3DParms cpyParams;
    memset(&cpyParams, 0, sizeof(cudaMemcpy3DParms));
    cpyParams.dstArray = volumeArray;
    cpyParams.extent = extent;
    cpyParams.kind = cudaMemcpyHostToDevice;
    cpyParams.srcPtr = make_cudaPitchedPtr(volumeData, dim.x * sizeof(char) * bytesPerElement, dim.x, dim.y);
    checkCudaErrors(cudaMemcpy3D(&cpyParams));

    // create texture
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = volumeArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    checkCudaErrors(cudaCreateTextureObject(&volumeTex, &resDesc, &texDesc, NULL));
}

void VolumeReader::CreateDeviceVolume(cudaVolume* volume)
{
    CreateTextures();

    glm::vec3 volumeExtent = GetVolumeSize();
    auto vmax = volumeExtent - volumeExtent * 0.5f;
    auto vmin = -vmax;
    cudaBBox bbox = cudaBBox(vmin, vmax);

    volume->Set(bbox, spacing, volumeTex);
    volume->SetInvMaxMagnitude(1.f / maxMagnitude);
}

glm::vec3 VolumeReader::GetVolumeSize()
{
    return glm::vec3(dim.x * spacing.x, dim.y * spacing.y, dim.z * spacing.z);
}

float VolumeReader::GetBoundingSphereRadius()
{
    auto size = GetVolumeSize();
    return glm::length(size) * 0.5f;
}

float VolumeReader::GetElementBoundingSphereRadius() const
{
    return glm::length(spacing) * 0.5f;
}
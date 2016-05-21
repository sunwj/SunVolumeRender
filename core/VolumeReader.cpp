//
// Created by 孙万捷 on 16/5/21.
//

#include <driver_types.h>
#include "VolumeReader.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "utils/std_image_write.h"

void VolumeReader::Read(std::string filename, std::string path)
{
    ClearHost();

    // image reader
    auto metaReader = vtkSmartPointer<vtkMetaImageReader>::New();
    metaReader->SetFileName((path + filename).c_str());
    metaReader->ReleaseDataFlagOn();
    metaReader->Update();

    auto imageData = metaReader->GetOutput();
    char* dataPtr = static_cast<char*>(imageData->GetScalarPointer());
    auto dataSpacing = metaReader->GetDataSpacing();
    auto dataDim = metaReader->GetDataExtent();     // data dimension structure [0, dimX - 1, 0, dimY - 1, 0, dimZ - 1]
    elementType = metaReader->GetDataScalarType();

    // rescale data
    auto size = (dataDim[1] + 1) * (dataDim[3] + 1) * (dataDim[5] + 1);
    switch(elementType)
    {
        case VTK_SHORT:
            data = new char[size * 2];
            Rescale<short, unsigned short>(reinterpret_cast<short*>(dataPtr), size);
            break;

        default:
            std::cerr<<"Unsupported data element type! Read volume data failed!"<<std::endl;
            exit(0);
            break;
    }

    this->spacing = glm::vec3(dataSpacing[0], dataSpacing[1], dataSpacing[2]);
    this->dim = glm::ivec3(dataDim[1] + 1, dataDim[3] + 1, dataDim[5] + 1);

    auto ptr = reinterpret_cast<unsigned short*>(data);
    unsigned char* tmp = new unsigned char[512 * 512 * 3];
    for(auto i = 0; i < 512; ++i)
    {
        for(auto j = 0; j < 512; ++j)
        {
            auto offset = i * 512 + j;
            tmp[offset * 3] = ptr[offset] / 65535.f * 255;
            tmp[offset * 3 + 1] = tmp[offset * 3];
            tmp[offset * 3 + 2] = tmp[offset * 3];
        }
    }

    stbi_write_bmp("test.bmp", 512, 512, 3, (char*)tmp);
    delete []tmp;
}

void VolumeReader::ClearHost()
{
    if(data != nullptr)
    {
        delete data;
        data = nullptr;
    }

    spacing = glm::vec3(0.f);
    dim = glm::ivec3(0);
}

void VolumeReader::ClearDevice()
{
    if(array != nullptr)
    {
        checkCudaErrors(cudaFreeArray(array));
        array = nullptr;
    }

    if(tex != 0)
    {
        checkCudaErrors(cudaDestroyTextureObject(tex));
        tex = 0;
    }
}

template <typename T, typename UT>
void VolumeReader::Rescale(T* dataPtr, size_t size)
{
    auto ptr1 = dataPtr;
    auto ptr2 = reinterpret_cast<UT*>(data);

    auto lowBound = std::numeric_limits<T>::max();
    auto upBound = std::numeric_limits<T>::min();
    for(auto i = 0; i < size; ++i)
    {
        upBound = std::max(upBound, dataPtr[i]);
        lowBound = std::min(lowBound, ptr1[i]);
    }
    auto extent = upBound - lowBound;

    auto dataTypeExtent = std::numeric_limits<UT>::max() - std::numeric_limits<UT>::min();
    // rescale data to unsigned data type
    for(auto i = 0; i < size; ++i)
    {
        ptr2[i] = (ptr1[i] - lowBound) / (float)extent * dataTypeExtent;
    }
}

void VolumeReader::CreateVolumeTexture()
{
    ClearDevice();

    int bytesPerElement = 0;
    switch(elementType)
    {
        case VTK_SHORT:
            bytesPerElement = 2;
            break;
        default:
            std::cout<<"Unsupported data element type! Create device volume failed!"<<std::endl;
            exit(0);
            break;
    }
    // allocate cudaArray
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(bytesPerElement * 8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaExtent extent = make_cudaExtent(dim.x, dim.y, dim.z);
    checkCudaErrors(cudaMalloc3DArray(&array, &channelDesc, extent, cudaArrayDefault));

    cudaMemcpy3DParms cpyParams;
    memset(&cpyParams, 0, sizeof(cudaMemcpy3DParms));
    cpyParams.dstArray = array;
    cpyParams.extent = extent;
    cpyParams.kind = cudaMemcpyHostToDevice;
    cpyParams.srcPtr = make_cudaPitchedPtr(data, dim.x * sizeof(char) * bytesPerElement, dim.x, dim.y);
    checkCudaErrors(cudaMemcpy3D(&cpyParams));

    // create texture
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    checkCudaErrors(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
}

void VolumeReader::CreateDeviceVolume(cudaVolume* volume)
{
    CreateVolumeTexture();

    glm::vec3 volumeExtent = GetVolumeSize();
    auto vmax = volumeExtent - volumeExtent * 0.5f;
    auto vmin = -vmax;
    cudaBBox bbox = cudaBBox(vmin, vmax);

    volume->Set(bbox, spacing, tex);
}

glm::vec3 VolumeReader::GetVolumeSize()
{
    return glm::vec3(dim.x * spacing.x, dim.y * spacing.y, dim.z * spacing.z);
}
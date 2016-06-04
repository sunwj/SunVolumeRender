//
// Created by 孙万捷 on 16/5/21.
//

#ifndef SUNVOLUMERENDER_VOLUMEREADER_H
#define SUNVOLUMERENDER_VOLUMEREADER_H

#include <vector>

#include <QFileInfo>

#include <vtkMetaImageReader.h>
#include <vtkImageData.h>
#include <vtkImageCast.h>
#include <vtkImageMagnitude.h>
#include <vtkImageAccumulate.h>
#include <vtkSmartPointer.h>
#include <vtkErrorCode.h>

#include <glm/glm.hpp>

#include <cuda_runtime.h>

#include "utils/helper_cuda.h"
#include "core/cuda_volume.h"

class VolumeReader
{
public:
    void Read(std::string filename);
    void CreateDeviceVolume(cudaVolume* volume);
    glm::vec3 GetVolumeSize();
    float GetBoundingSphereRadius();

private:
    void ClearHost();
    void ClearDevice();
    template <typename T, typename UT>
    void Rescale(T* dataPtr, size_t size, float dataMin, float dataMax);
    void CreateTextures();

public:
    std::vector<uint32_t> histogram;

private:
    glm::vec3 spacing = glm::vec3(glm::uninitialize);
    glm::ivec3 dim = glm::ivec3(glm::uninitialize);
    char* volumeData = nullptr;

    cudaArray* volumeArray = nullptr;
    cudaTextureObject_t volumeTex = 0;
};


#endif //SUNVOLUMERENDER_VOLUMEREADER_H

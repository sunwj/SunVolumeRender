//
// Created by 孙万捷 on 16/3/4.
//

#ifndef SUNVOLUMERENDER_PATHTRACER_H
#define SUNVOLUMERENDER_PATHTRACER_H

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include "core/cuda_camera.h"
#include "core/cuda_bbox.h"
#include "core/cuda_transfer_function.h"

extern "C" void rendering(glm::u8vec4* img, const cudaBBox& volumeBox, const cudaCamera& camera, unsigned int frameNo);

extern "C" void setup_transferfunction(const cudaTransferFunction& tf);

#endif //SUNVOLUMERENDER_PATHTRACER_H

//
// Created by 孙万捷 on 16/7/6.
//

#ifndef SUNVOLUMERENDER_RAYCASTING_H_H
#define SUNVOLUMERENDER_RAYCASTING_H_H

extern "C" void render_raycasting(glm::u8vec4* img, cudaVolume& volume, cudaTransferFunction& transferFunction, cudaCamera& camera, float stepSize);

#endif //SUNVOLUMERENDER_RAYCASTING_H_H

//
// Created by 孙万捷 on 16/3/4.
//

#ifndef SUNVOLUMERENDER_CANVAS_H
#define SUNVOLUMERENDER_CANVAS_H

#include <QGLWidget>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QTimerEvent>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "core/VolumeReader.h"
#include "core/lights/lights.h"

#include "common.h"
#include "utils/helper_cuda.h"
#include "pathtracer.h"
#include "raycasting.h"
#include "core/cuda_transfer_function.h"
#include "core/cuda_volume.h"
#include "core/render_parameters.h"

enum RenderMode {RENDER_MODE_PATHTRACER, RENDER_MODE_RAYCASTING};

class Canvas : public QGLWidget
{
    Q_OBJECT
public:
    explicit Canvas(const QGLFormat& format, QWidget* parent = 0);
    virtual ~Canvas();

    void LoadVolume(std::string filename);
    void StartTimer() {timerId = this->startTimer(0);}
    void KillTimer() {this->killTimer(timerId);}

    void ReStartRender()
    {
        updateGL();
        renderParams.frameNo = 0;
    }

    void SetTransferFunction(const cudaTextureObject_t& tex, float maxOpacity)
    {
        transferFunction.Set(tex, maxOpacity);
        setup_transferfunction(transferFunction);
        ReStartRender();
    }

    void SetDensityScale(double s)
    {
        deviceVolume.SetDensityScale(s);
        setup_volume(deviceVolume);
        ReStartRender();
    }

    void SetScatterTimes(double val)
    {
        renderParams.traceDepth = val;
        ReStartRender();
    }

    void SetRenderMode(RenderMode mode)
    {
        if(mode == RENDER_MODE_PATHTRACER)
        {
            StartTimer();
        }
        else if(mode == RENDER_MODE_RAYCASTING)
        {
            KillTimer();
        }

        renderMode = mode;
        ReStartRender();
    }

    // lights
    void SetEnvLightBackground(const glm::vec3& color)
    {
        lights.SetEnvionmentLight(color);
        setup_env_lights(lights.environmentLight);
        ReStartRender();
    }

    void SetEnvLightMap(std::string filename)
    {
        lights.SetEnvironmentLight(filename);
        setup_env_lights(lights.environmentLight);
        ReStartRender();
    }

    void SetEnvLightOffset(const glm::vec2& offset)
    {
        lights.SetEnvironmentLightOffset(offset);
        setup_env_lights(lights.environmentLight);
        ReStartRender();
    }

    void SetEnvLightIntensity(float intensity)
    {
        lights.SetEnvironmentLightIntensity(intensity);
        setup_env_lights(lights.environmentLight);
        ReStartRender();
    }

    void SetAreaLights()
    {
        setup_area_lights(lights.areaLights.data(), lights.areaLights.size());
        ReStartRender();
    }

    // camera
    void SetFOV(float fov)
    {
        this->fov = fov;
        UpdateCamera();
        ReStartRender();
    }

    void SetApeture(float apeture)
    {
        this->apeture = apeture;
        UpdateCamera();
        ReStartRender();
    }

    void SetFocalLength(float focalLength)
    {
        this->focalLength = focalLength;
        UpdateCamera();
        ReStartRender();
    }

    void SetExposure(float exposure)
    {
        this->exposure = exposure;
        UpdateCamera();
        ReStartRender();
    }

    // clip plane
    void SetXClipPlane(double min, double max)
    {
        deviceVolume.SetXClipPlane(glm::vec2(float(min), float(max)));
        setup_volume(deviceVolume);
        ReStartRender();
    }

    void SetYClipPlane(double min, double max)
    {
        deviceVolume.SetYClipPlane(glm::vec2(float(min), float(max)));
        setup_volume(deviceVolume);
        ReStartRender();
    }

    void SetZClipPlane(double min, double max)
    {
        deviceVolume.SetZClipPlane(glm::vec2(float(min), float(max)));
        setup_volume(deviceVolume);
        ReStartRender();
    }

protected:
    //opengl
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();

    //mouse
    QPointF PixelPosToViewPos(const QPointF& pt)
    {
        return QPointF(2.f * static_cast<float>(pt.x()) / WIDTH - 1.f,
        1.f - 2.f * static_cast<float>(pt.y()) / HEIGHT);
    }
    void mousePressEvent(QMouseEvent* e);
    void mouseReleaseEvent(QMouseEvent* e);
    void mouseMoveEvent(QMouseEvent* e);
    void wheelEvent(QWheelEvent* e);

    // timer
    void timerEvent(QTimerEvent* e) {this->update();}

    //others
    void ZoomToExtent();

private:
    void UpdateCamera();

public:
    VolumeReader volumeReader;
    Lights lights;

private:
    int timerId;
    bool ready = false;

    GLuint pbo = 0;
    cudaGraphicsResource* resource;
    glm::u8vec4* img;
    QPointF mouseStartPoint;
    float exposure = 1.f;
    float apeture = 0.f;
    float fov = 45.f;
    float focalLength = 1.f;
    float eyeDist = 0.f;
    glm::vec2 cameraTranslate = glm::vec2(0.f);
    glm::mat4 viewMat = glm::mat4(1.f);

    RenderParams renderParams;

    cudaCamera camera;
    cudaVolume deviceVolume;
    cudaTransferFunction transferFunction;

    RenderMode renderMode = RENDER_MODE_RAYCASTING;
};


#endif //SUNVOLUMERENDER_CANVAS_H

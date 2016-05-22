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

#include "common.h"
#include "utils/helper_cuda.h"
#include "core/pathtracer.h"
#include "core/cuda_transfer_function.h"
#include "core/cuda_volume.h"
#include "core/render_parameters.h"

class Canvas : public QGLWidget
{
    Q_OBJECT
public:
    explicit Canvas(const QGLFormat& format, QWidget* parent = 0);
    virtual ~Canvas();

    void SetTransferFunction(const cudaTextureObject_t& tex, float maxOpacity)
    {
        transferFunction.Set(tex);
        setup_transferfunction(transferFunction);

        ReStartRender();
    };

    void ReStartRender() {renderParams.frameNo = 0;}

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

private:
    GLuint pbo = 0;
    cudaGraphicsResource* resource;
    glm::u8vec4* img;
    QPointF mouseStartPoint;
    float fov = 45.f;
    float eyeDist = 0.f;
    glm::vec2 cameraTranslate = glm::vec2(0.f);
    glm::mat4 viewMat = glm::mat4(1.f);

    VolumeReader volumeReader;
    RenderParams renderParams;

    cudaCamera camera;
    cudaVolume deviceVolume;
    cudaTransferFunction transferFunction;
};


#endif //SUNVOLUMERENDER_CANVAS_H

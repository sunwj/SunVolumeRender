#ifndef TRANSFERFUNCTION_H
#define TRANSFERFUNCTION_H

// STD include
#include <iostream>

// Qt include
#include <QObject>
#include <QSharedPointer>
#include <QFileDialog>

// CTK include
#include <ctkTransferFunction.h>
#include <ctkVTKPiecewiseFunction.h>
#include <ctkVTKColorTransferFunction.h>
#include <ctkVTKCompositeFunction.h>

// VTK include
#include <vtkSmartPointer.h>
#include <vtkPiecewiseFunction.h>
#include <vtkColorTransferFunction.h>

// cuda include
#include <cuda.h>
#include <cuda_runtime.h>

#include "utils/helper_cuda.h"

#define TABLE_SIZE 1024
class TransferFunction : public QObject
{
    Q_OBJECT
public:
    explicit TransferFunction(vtkSmartPointer<vtkPiecewiseFunction> otf, vtkSmartPointer<vtkColorTransferFunction> ctf, QObject *parent = 0);
    ~TransferFunction();

    cudaTextureObject_t GetCompositeTFTextureObject() {return compositeTex;}
    void SaveCurrentTFConfiguration();
    void LoadExistingTFConfiguration();

signals:
    void Changed();

public slots:

protected slots:
    void onOpacityTFChanged();
    void onColorTFChanged();

private:
    QSharedPointer<ctkTransferFunction> otf;
    QSharedPointer<ctkTransferFunction> ctf;

    vtkSmartPointer<vtkPiecewiseFunction> opacityTF;
    vtkSmartPointer<vtkColorTransferFunction> colorTF;

    float opacityTable[TABLE_SIZE];
    float colorTable[TABLE_SIZE * 3];
    float compositeTable[TABLE_SIZE * 4];

    cudaArray *array;
    cudaChannelFormatDesc channelDesc;
    cudaResourceDesc resourceDesc;
    cudaTextureDesc texDesc;
    cudaTextureObject_t compositeTex;
};

#endif // TRANSFERFUNCTION_H

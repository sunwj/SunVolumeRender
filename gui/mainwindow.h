#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>

#include <vtkVolumeProperty.h>

#include <ctkVTKScalarsToColorsView.h>

#include "qcustomplot.h"
#include "AddAreaLightDialog.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "gui/transferfunction.h"
#include "gui/canvas.h"
#include "common.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    void ConfigureTransferFunction();
    void ConfigureCanvas();
    void ConfigureActions();
    void ConfigureLight();

private slots:
    void onTransferFunctionChanged();

    void onEnvLightUOffsetChanged(double u);
    void onEnvLightVOffsetChanged(double v);
    void onEnvLightBackgroundChanged(QColor color);
    void onEnvLightMapChanged(const QString& path);
    void onEnvLightIntensityChanged(double intensity);

    void onAreaLightSelected();
    void onAddAreaLight();
    void onRemoveLight();
    void onAreaLightColorChanged(QColor color);
    void onAreaLightIntensityChanged(double val);
    void onAreaLightRadiusChanged(double val);
    void onAreaLightDistanceChanged(double val);
    void onAreaLightLatitudeChanged(double val);
    void onAreaLightLongitudeChanged(double val);

    void onFileOpen();


private:
    Ui::MainWindow *ui;

    TransferFunction* tf;
    Canvas* canvas;
};

#endif // MAINWINDOW_H

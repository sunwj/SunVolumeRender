#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <vtkVolumeProperty.h>

#include <ctkVTKScalarsToColorsView.h>

#include "qcustomplot.h"

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
    void onTransferFunctionChanged()
    {
        canvas->SetTransferFunction(tf->GetCompositeTFTextureObject(), tf->GetMaxOpacityValue());
    };

    void onEnvLightUOffsetChanged(double u);
    void onEnvLightVOffsetChanged(double v);
    void onEnvLightBackgroundChanged(QColor color);
    void onEnvLightMapChanged(const QString& path);

    void onFileOpen();


private:
    Ui::MainWindow *ui;

    TransferFunction* tf;
    Canvas* canvas;
};

#endif // MAINWINDOW_H

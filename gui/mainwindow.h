#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <vtkVolumeProperty.h>

#include <ctkVTKScalarsToColorsView.h>

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

private slots:
    void onTransferFunctionChanged()
    {
        //std::cerr<<"Transfer function changed!"<<std::endl;
        canvas->SetTransferFunctionTexture(tf->GetCompositeTFTextureObject());
    };

private:
    Ui::MainWindow *ui;

    TransferFunction* tf;
    Canvas* canvas;
};

#endif // MAINWINDOW_H

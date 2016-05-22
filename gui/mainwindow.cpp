#include <vtkSmartPointer.h>
#include <vtkPiecewiseFunction.h>
#include <vtkColorTransferFunction.h>

#include "gui/mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->dockWidget->setTitleBarWidget(new QWidget);

    ConfigureTransferFunction();
    ConfigureCanvas();

    // initialize transferfunction on device
    canvas->SetTransferFunction(this->tf->GetCompositeTFTextureObject(), this->tf->GetMaxOpacityValue());
}

MainWindow::~MainWindow()
{
    delete ui;
    exit(0);
}

void MainWindow::ConfigureTransferFunction()
{
    vtkSmartPointer<vtkPiecewiseFunction> opacityTransferFunc = vtkSmartPointer<vtkPiecewiseFunction>::New();
    vtkSmartPointer<vtkColorTransferFunction> colorTransferFunc = vtkSmartPointer<vtkColorTransferFunction>::New();

    opacityTransferFunc->AddPoint(0, 0.0, 0.5, 0.5);
    for(int i = 1; i <= 10; ++i)
    {
        opacityTransferFunc->AddPoint(0.1 * i, 0.5, 0.5, 0.5);
    }

    colorTransferFunc->AddRGBPoint(0. , 69./255., 199./255.,   186./255.);
    colorTransferFunc->AddRGBPoint(0.2,  172./255., 3./255., 57./255.);
    colorTransferFunc->AddRGBPoint(0.4,  169./255., 83./255., 58./255.);
    colorTransferFunc->AddRGBPoint(0.6,  43./255., 32./255.,  161./255.);
    colorTransferFunc->AddRGBPoint(0.8,  247./255., 158./255., 97./255.);
    colorTransferFunc->AddRGBPoint(1.,  183./255., 7./255., 140./255.);

    tf = new TransferFunction(opacityTransferFunc, colorTransferFunc);

    ui->opacityTransferFunc->view()->addCompositeFunction(colorTransferFunc, opacityTransferFunc, false, true);
    ui->colorTransferFunc->view()->addColorTransferFunction(colorTransferFunc);

    ui->opacityTransferFunc->view()->setAxesToChartBounds();
    ui->colorTransferFunc->view()->setAxesToChartBounds();

    connect(tf, SIGNAL(Changed()), this, SLOT(onTransferFunctionChanged()));
}

void MainWindow::ConfigureCanvas()
{
    QGLFormat format;
    format.setDoubleBuffer(true);
    format.setRgba(true);
    format.setDepth(true);

    canvas = new Canvas(format, this);
    canvas->setMinimumSize(WIDTH, HEIGHT);
    canvas->setMaximumSize(WIDTH, HEIGHT);

    ui->centralLayout->addWidget(canvas);
}

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
    ConfigureActions();
    ConfigureLight();

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

void MainWindow::ConfigureActions()
{
    connect(ui->actionLoadVolume, SIGNAL(triggered()), this, SLOT(onFileOpen()));
}

void MainWindow::onFileOpen()
{
    QString fileName = "";
    QFileDialog dlg(this, tr("Load volume data"), "./", tr("All(*.*)"));
    if(dlg.exec())
        fileName = dlg.selectedFiles()[0];

    canvas->LoadVolume(fileName.toStdString());

    const std::vector<uint32_t >& histogram = canvas->GetVolume().histogram;
    QVector<double> xAxis(static_cast<uint32_t>(histogram.size()));
    QVector<double> yAxis(static_cast<uint32_t>(histogram.size()));
    for(auto i = 0; i < static_cast<uint32_t>(histogram.size()); ++i)
    {
        xAxis[i] = i;
        yAxis[i] = log10(histogram[i] + 1);
    }

    ui->histogramChart->addGraph();
    ui->histogramChart->graph(0)->setPen(QPen(QColor(255, 0, 0)));
    ui->histogramChart->graph(0)->setBrush(QBrush(QColor(255, 0, 0)));
    ui->histogramChart->graph(0)->setData(xAxis, yAxis);
    ui->histogramChart->graph(0)->rescaleAxes();
    ui->histogramChart->replot();
}

void MainWindow::ConfigureLight()
{
    connect(ui->SliderWidget_EnvLightUOffset, SIGNAL(valueChanged(double)), this, SLOT(onEnvLightUOffsetChanged(double)));
    connect(ui->SliderWidget_EnvLightVOffset, SIGNAL(valueChanged(double)), this, SLOT(onEnvLightVOffsetChanged(double)));
    connect(ui->ColorPickerButton_EnvBackground, SIGNAL(colorChanged(QColor)), this, SLOT(onEnvLightBackgroundChanged(QColor)));

    QStringList nameFilters;
    nameFilters << "*.hdr";
    ui->PathLineEdit_EnvMap->setNameFilters(nameFilters);
    connect(ui->PathLineEdit_EnvMap, SIGNAL(currentPathChanged(const QString&)), this, SLOT(onEnvLightMapChanged(const QString&)));
}

void MainWindow::onEnvLightUOffsetChanged(double u)
{
    double v = ui->SliderWidget_EnvLightVOffset->value();
    canvas->SetEnvLightOffset(glm::vec2(u, v));
}

void MainWindow::onEnvLightVOffsetChanged(double v)
{
    double u = ui->SliderWidget_EnvLightUOffset->value();
    canvas->SetEnvLightOffset(glm::vec2(u, v));
}

void MainWindow::onEnvLightBackgroundChanged(QColor color)
{
    canvas->SetEnvLightBackground(glm::vec3(color.red(), color.green(), color.blue()) * 0.002f);
}

void MainWindow::onEnvLightMapChanged(const QString &path)
{
    canvas->SetEnvLightMap(path.toStdString());
}

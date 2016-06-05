#include <vtkSmartPointer.h>
#include <vtkPiecewiseFunction.h>
#include <vtkColorTransferFunction.h>
#include <QtGui/QMessageBox>

#include "gui/mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->dockWidget->setTitleBarWidget(new QWidget);
    ui->tabController->setEnabled(false);

    ConfigureTransferFunction();
    ConfigureCanvas();
    ConfigureActions();
    ConfigureLight();
    ConfigureCamera();

    // initialize transferfunction on device
    canvas->SetTransferFunction(this->tf->GetCompositeTFTextureObject(), 0.5f);
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

    ui->opacityTransferFunc->addExtraWidget(new QLabel(tr("Opacity")));
    ui->colorTransferFunc->addExtraWidget(new QLabel(tr("Diffuse Color")));
    ui->opacityTransferFunc->view()->addCompositeFunction(colorTransferFunc, opacityTransferFunc, false, true);
    ui->colorTransferFunc->view()->addColorTransferFunction(colorTransferFunc);

    ui->opacityTransferFunc->view()->setAxesToChartBounds();
    ui->colorTransferFunc->view()->setAxesToChartBounds();

    connect(tf, SIGNAL(Changed()), this, SLOT(onTransferFunctionChanged()));
}

void MainWindow::onTransferFunctionChanged()
{
    canvas->SetTransferFunction(tf->GetCompositeTFTextureObject(), tf->GetMaxOpacityValue());
};

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

    connect(ui->SliderWidget_scatterTimes, SIGNAL(valueChanged(double)), this, SLOT(onScatterTimesChanged(double)));
}

void MainWindow::ConfigureActions()
{
    connect(ui->actionLoadVolume, SIGNAL(triggered()), this, SLOT(onFileOpen()));
}

void MainWindow::ConfigureLight()
{
    // arealight
    connect(ui->toolButton_addAreaLight, SIGNAL(clicked()), this, SLOT(onAddAreaLight()));
    connect(ui->toolButton_removeAreaLight, SIGNAL(clicked()), this, SLOT(onRemoveLight()));
    connect(ui->listWidget_areaLights, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(onAreaLightSelected()));

    connect(ui->ColorPickerButton_areaLightColor, SIGNAL(colorChanged(QColor)), this, SLOT(onAreaLightColorChanged(QColor)));
    connect(ui->SliderWidget_areaLightIntensity, SIGNAL(valueChanged(double)), this, SLOT(onAreaLightIntensityChanged(double)));
    connect(ui->SliderWidget_areaLightRadius, SIGNAL(valueChanged(double)), this, SLOT(onAreaLightRadiusChanged(double)));
    connect(ui->SliderWidget_areaLightDistance, SIGNAL(valueChanged(double)), this, SLOT(onAreaLightDistanceChanged(double)));
    connect(ui->SliderWidget_areaLightLatitude, SIGNAL(valueChanged(double)), this, SLOT(onAreaLightLatitudeChanged(double)));
    connect(ui->SliderWidget_areaLightLongitude, SIGNAL(valueChanged(double)), this, SLOT(onAreaLightLongitudeChanged(double)));

    // environment
    connect(ui->SliderWidget_EnvLightUOffset, SIGNAL(valueChanged(double)), this, SLOT(onEnvLightUOffsetChanged(double)));
    connect(ui->SliderWidget_EnvLightVOffset, SIGNAL(valueChanged(double)), this, SLOT(onEnvLightVOffsetChanged(double)));
    connect(ui->ColorPickerButton_EnvBackground, SIGNAL(colorChanged(QColor)), this, SLOT(onEnvLightBackgroundChanged(QColor)));
    connect(ui->SliderWidget_EnvLightIntensity, SIGNAL(valueChanged(double)), this, SLOT(onEnvLightIntensityChanged(double)));

    QStringList nameFilters;
    nameFilters << "*.hdr";
    ui->PathLineEdit_EnvMap->setNameFilters(nameFilters);
    connect(ui->PathLineEdit_EnvMap, SIGNAL(currentPathChanged(const QString&)), this, SLOT(onEnvLightMapChanged(const QString&)));
}

void MainWindow::ConfigureCamera()
{
    connect(ui->SliderWidget_FOV, SIGNAL(valueChanged(double)), this, SLOT(onCameraFOVChanged(double)));
    connect(ui->SliderWidget_focalLength, SIGNAL(valueChanged(double)), this, SLOT(onCameraFocalLengthChanged(double)));
    connect(ui->SliderWidget_exposure, SIGNAL(valueChanged(double)), this, SLOT(onCameraExposureChaned(double)));
    connect(ui->SliderWidget_apeture, SIGNAL(valueChanged(double)), this, SLOT(onCameraApetureChanged(double)));
}

void MainWindow::onFileOpen()
{
    QString fileName = "";
    QFileDialog dlg(this, tr("Load volume data"), "./", tr("Volume Data(*.mhd *.mha)"));
    if(dlg.exec())
        fileName = dlg.selectedFiles()[0];
    if(fileName.isEmpty()) return;

    canvas->LoadVolume(fileName.toStdString());

    const std::vector<uint32_t >& histogram = canvas->volumeReader.histogram;
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

    ui->tabController->setEnabled(true);
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
    canvas->SetEnvLightBackground(glm::vec3(color.red() / 255.f, color.green() / 255.f, color.blue() / 255.f));
}

void MainWindow::onEnvLightIntensityChanged(double intensity)
{
    canvas->SetEnvLightIntensity(intensity);
}

void MainWindow::onEnvLightMapChanged(const QString &path)
{
    canvas->SetEnvLightMap(path.toStdString());
}

void MainWindow::onAddAreaLight()
{
    if(ui->listWidget_areaLights->count() == MAX_LIGHT_SOURCES)
    {
        QMessageBox::warning(this, tr("Warning"), tr("Exceed number of light sources allowed"), QMessageBox::Ok);
        return;
    }

    AddAreaLightDialog dialog;
    if(dialog.exec() == 1)
    {
        QString lightName = dialog.GetLightName();
        if(lightName.isEmpty())
        {
            QMessageBox::critical(this, tr("Error"), tr("light name cannot be empty!"), QMessageBox::Ok);
            return;
        }

        QListWidgetItem* item = new QListWidgetItem(QIcon(":/icons/light_bulb"), lightName);
        ui->listWidget_areaLights->addItem(item);

        // initialize and add light parameters to Lights
        cudaAreaLight areaLight;
        areaLight.Set(cudaDisk(glm::vec3(0.f), glm::vec3(0.f, -1.f, 0.f), 10.f), glm::vec3(1.f), 500.f);
        canvas->lights.AddAreaLights(areaLight, glm::vec3(0.f, 0.f, canvas->volumeReader.GetBoundingSphereRadius() * 1.5f + 1.f));

        glm::mat4 mat(1.f);
        mat = glm::translate(mat, -(canvas->volumeReader.GetBoundingSphereRadius() * 1.5f + 1.f) * glm::vec3(0.f, -1.f, 0.f));
        glm::vec3 pos = glm::vec3(mat * glm::vec4(0.f, 0.f, 0.f, 1.f));

        auto light = canvas->lights.GetLastAreaLight();
        light->SetPosition(pos);
        light->SetNormal(glm::normalize(-pos));

        canvas->SetAreaLights();
    }
}

void MainWindow::onRemoveLight()
{
    if(ui->listWidget_areaLights->count() == 0)
    {
        QMessageBox::critical(this, tr("Error"), tr("No light to remove"), QMessageBox::Ok);
        return;
    }

    if(ui->listWidget_areaLights->currentRow() == -1)
    {
        QMessageBox::warning(this, tr("Warning"), tr("Select a light first"), QMessageBox::Ok);
        return;
    }

    int idx = ui->listWidget_areaLights->currentRow();

    delete ui->listWidget_areaLights->item(idx);
    ui->listWidget_areaLights->update();
    canvas->lights.RemoveLights(idx);

    canvas->SetAreaLights();
}

void MainWindow::onAreaLightSelected()
{
    int idx = ui->listWidget_areaLights->currentRow();

    auto light = canvas->lights.GetAreaLight(idx);
    auto transformation = canvas->lights.GetAreaLightTransformation(idx);

    auto color = light->GetColor();
    ui->ColorPickerButton_areaLightColor->setColor(QColor(color.x * 255, color.y * 255, color.z * 255));

    auto intensity = light->GetIntensity();
    ui->SliderWidget_areaLightIntensity->setValue(intensity);

    auto radius = light->GetRadius();
    ui->SliderWidget_areaLightRadius->setValue(radius);

    auto distance = transformation->z;
    ui->SliderWidget_areaLightDistance->setValue(distance);

    auto latitude = transformation->x;
    ui->SliderWidget_areaLightLatitude->setValue(latitude);

    auto longitude = transformation->y;
    ui->SliderWidget_areaLightLongitude->setValue(longitude);
}

void MainWindow::onAreaLightColorChanged(QColor color)
{
    int idx = ui->listWidget_areaLights->currentRow();
    if(idx == -1)
    {
        QMessageBox::warning(this, tr("Warning"), tr("Select a light first"), QMessageBox::Ok);
        return;
    }

    auto light = canvas->lights.GetAreaLight(idx);
    light->SetColor(glm::vec3(color.red() / 255.f, color.green() / 255.f, color.blue() / 255.f));

    canvas->SetAreaLights();
}

void MainWindow::onAreaLightIntensityChanged(double val)
{
    int idx = ui->listWidget_areaLights->currentRow();
    if(idx == -1)
    {
        QMessageBox::warning(this, tr("Warning"), tr("Select a light first"), QMessageBox::Ok);
        return;
    }

    auto light = canvas->lights.GetAreaLight(idx);
    light->SetIntensity(val);

    canvas->SetAreaLights();
}

void MainWindow::onAreaLightRadiusChanged(double val)
{
    int idx = ui->listWidget_areaLights->currentRow();
    if(idx == -1)
    {
        QMessageBox::warning(this, tr("Warning"), tr("Select a light first"), QMessageBox::Ok);
        return;
    }

    auto light = canvas->lights.GetAreaLight(idx);
    light->SetRadius(val);

    canvas->SetAreaLights();
}

void MainWindow::onAreaLightDistanceChanged(double val)
{
    int idx = ui->listWidget_areaLights->currentRow();
    if(idx == -1)
    {
        QMessageBox::warning(this, tr("Warning"), tr("Select a light first"), QMessageBox::Ok);
        return;
    }

    auto transformation = canvas->lights.GetAreaLightTransformation(idx);
    auto light = canvas->lights.GetAreaLight(idx);
    transformation->z = val;

    glm::mat4 mat(1.f);
    mat = glm::translate(mat, -transformation->z * glm::vec3(0.f, -1.f, 0.f));
    glm::vec3 pos = glm::vec3(mat * glm::vec4(0.f, 0.f, 0.f, 1.f));

    auto mat_rx = glm::rotate(glm::radians(transformation->x), glm::vec3(1.f, 0.f, 0.f));
    auto mat_ry = glm::rotate(glm::radians(transformation->y), glm::vec3(0.f, 0.f, 1.f));
    pos = glm::vec3(mat_rx * mat_ry * glm::vec4(pos, 1.f));

    light->SetPosition(pos);
    light->SetNormal(glm::normalize(-pos));

    canvas->SetAreaLights();
}

void MainWindow::onAreaLightLatitudeChanged(double val)
{
    int idx = ui->listWidget_areaLights->currentRow();
    if(idx == -1)
    {
        QMessageBox::warning(this, tr("Warning"), tr("Select a light first"), QMessageBox::Ok);
        return;
    }

    auto transformation = canvas->lights.GetAreaLightTransformation(idx);
    auto light = canvas->lights.GetAreaLight(idx);
    transformation->x = val;

    glm::mat4 mat(1.f);
    mat = glm::translate(mat, -transformation->z * glm::vec3(0.f, -1.f, 0.f));
    glm::vec3 pos = glm::vec3(mat * glm::vec4(0.f, 0.f, 0.f, 1.f));

    auto mat_rx = glm::rotate(glm::radians(transformation->x), glm::vec3(1.f, 0.f, 0.f));
    auto mat_ry = glm::rotate(glm::radians(transformation->y), glm::vec3(0.f, 0.f, 1.f));
    pos = glm::vec3(mat_rx * mat_ry * glm::vec4(pos, 1.f));

    light->SetPosition(pos);
    light->SetNormal(glm::normalize(-pos));

    canvas->SetAreaLights();
}

void MainWindow::onAreaLightLongitudeChanged(double val)
{
    int idx = ui->listWidget_areaLights->currentRow();
    if(idx == -1)
    {
        QMessageBox::warning(this, tr("Warning"), tr("Select a light first"), QMessageBox::Ok);
        return;
    }

    auto transformation = canvas->lights.GetAreaLightTransformation(idx);
    auto light = canvas->lights.GetAreaLight(idx);
    transformation->y = val;

    glm::mat4 mat(1.f);
    mat = glm::translate(mat, -transformation->z * glm::vec3(0.f, -1.f, 0.f));
    glm::vec3 pos = glm::vec3(mat * glm::vec4(0.f, 0.f, 0.f, 1.f));

    auto mat_rx = glm::rotate(glm::radians(transformation->x), glm::vec3(1.f, 0.f, 0.f));
    auto mat_ry = glm::rotate(glm::radians(transformation->y), glm::vec3(0.f, 0.f, 1.f));
    pos = glm::vec3(mat_rx * mat_ry * glm::vec4(pos, 1.f));

    light->SetPosition(pos);
    light->SetNormal(glm::normalize(-pos));

    canvas->SetAreaLights();
}

void MainWindow::onCameraFOVChanged(double val)
{
    canvas->SetFOV(val);
}

void MainWindow::onCameraFocalLengthChanged(double val)
{
    canvas->SetFocalLength(val);
}

void MainWindow::onCameraExposureChaned(double val)
{
    canvas->SetExposure(val);
}

void MainWindow::onCameraApetureChanged(double val)
{
    canvas->SetApeture(val);
}

void MainWindow::onScatterTimesChanged(double val)
{
    canvas->SetScatterTimes(val);
}

#include "gui/mainwindow.h"
#include <QApplication>
#include <qtextstream.h>
#include <qfile.h>

void chooseBestDevice()
{
    // choose the best device as the current device
    int num_devices = 0;
    int maxComputeCapability = 0;
    checkCudaErrors(cudaGetDeviceCount(&num_devices));
    printf("%d devices found on this platform:\n", num_devices);

    int choice = 0;
    for(int i = 0; i < num_devices; ++i)
    {
        cudaDeviceProp property;
        checkCudaErrors(cudaGetDeviceProperties(&property, i));

        char *name = property.name;
        int computeCapability = property.major * 10 + property.minor;
        printf("%d Device name: %s\t Compute capability: %d.%d\n", i, name, property.major, property.minor);

        choice = maxComputeCapability > computeCapability ? choice : i;
        maxComputeCapability = maxComputeCapability > computeCapability ? maxComputeCapability : computeCapability;
    }

    printf("Choice device %d\n", choice);
    fflush(stdout);

    checkCudaErrors(cudaSetDevice(choice));
}

int main(int argc, char *argv[])
{
    chooseBestDevice();
    QApplication a(argc, argv);

    // load stylesheet
    QFile f(":qdarkstyle/style.qss");
    if (!f.exists())
    {
        printf("Unable to set stylesheet, file not found\n");
    }
    else
    {
        f.open(QFile::ReadOnly | QFile::Text);
        QTextStream ts(&f);
        a.setStyleSheet(ts.readAll());
    }

    MainWindow w;
    w.show();

    return a.exec();
}

#include "transferfunction.h"

TransferFunction::TransferFunction(vtkSmartPointer<vtkPiecewiseFunction> otf, vtkSmartPointer<vtkColorTransferFunction> ctf, QObject *parent) : QObject(parent)
{
    opacityTF = otf;
    colorTF = ctf;

    this->otf = QSharedPointer<ctkTransferFunction>(new ctkVTKPiecewiseFunction(opacityTF));
    this->ctf = QSharedPointer<ctkTransferFunction>(new ctkVTKColorTransferFunction(colorTF));

    connect(this->otf.data(), SIGNAL(changed()), this, SLOT(onOpacityTFChanged()));
    connect(this->ctf.data(), SIGNAL(changed()), this, SLOT(onColorTFChanged()));

    compositeTex = 0;

    // initialize each table
    opacityTF->GetTable(0.0, 1.0, TABLE_SIZE, opacityTable);
    colorTF->GetTable(0.0, 1.0, TABLE_SIZE, colorTable);
    size_t j = 0, k = 0, m = 0;
    for(auto i = 0; i < TABLE_SIZE; ++i)
    {
        compositeTable[j++] = colorTable[k++];
        compositeTable[j++] = colorTable[k++];
        compositeTable[j++] = colorTable[k++];

        maxOpacity = fmaxf(maxOpacity, opacityTable[m]);
        compositeTable[j++] = opacityTable[m++];
    }

    channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaMallocArray(&array, &channelDesc, TABLE_SIZE));
    checkCudaErrors(cudaMemcpyToArray(array, 0, 0, compositeTable, sizeof(float) * TABLE_SIZE * 4, cudaMemcpyHostToDevice));

    memset(&resourceDesc, 0, sizeof(resourceDesc));
    resourceDesc.resType = cudaResourceTypeArray;
    resourceDesc.res.array.array = array;

    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.normalizedCoords = true;
    texDesc.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&compositeTex, &resourceDesc, &texDesc, NULL));
}

TransferFunction::~TransferFunction()
{
    if(compositeTex)
        checkCudaErrors(cudaDestroyTextureObject(compositeTex));

    checkCudaErrors(cudaFreeArray(array));
}

void TransferFunction::SaveCurrentTFConfiguration()
{
    QString filename = QFileDialog::getSaveFileName(0, tr("Save transfer function"), QDir::currentPath(), tr("TF Files (*.tf)"));
    if(filename.isEmpty())
    {
        return;
    }

    std::ofstream output(filename.toStdString().c_str(), std::ios_base::binary | std::ios_base::trunc);
    if(!output)
    {
        std::cerr<<"unable to open file"<<std::endl;
        exit(0);
    }

    int size = opacityTF->GetSize();
    output.write((char*)&size, sizeof(int));
    for(int i = 0; i < size; ++i)
    {
        double val[4] = {0};
        opacityTF->GetNodeValue(i, val);
        output.write((char*)val, sizeof(double) * 4);
    }

    size = colorTF->GetSize();
    output.write((char*)&size, sizeof(int));
    for(int i = 0; i < size; ++i)
    {
        double val[6] = {0};
        colorTF->GetNodeValue(i, val);
        output.write((char*)val, sizeof(double) * 6);
    }

    output.close();
}

void TransferFunction::LoadExistingTFConfiguration()
{
    QString filename = QFileDialog::getOpenFileName(0, tr("Load transfer function"), QDir::currentPath(), tr("TF Files (*.tf)"));
    if(filename.isEmpty())
    {
        return;
    }

    std::ifstream input(filename.toStdString().c_str(), std::ios_base::binary);
    if(!input)
    {
        std::cerr<<"unable to open file"<<std::endl;
        exit(0);
    }

    int size = 0;
    input.read((char*)&size, sizeof(int));
    opacityTF->RemoveAllPoints();
    for(int i = 0; i < size; ++i)
    {
        double val[4] = {0};
        input.read((char*)val, sizeof(double) * 4);
        opacityTF->AddPoint(val[0], val[1], val[2], val[3]);
    }

    input.read((char*)&size, sizeof(int));
    colorTF->RemoveAllPoints();
    for(int i = 0; i < size; ++i)
    {
        double val[6] = {0};
        input.read((char*)val, sizeof(double) * 6);
        colorTF->AddRGBPoint(val[0], val[1], val[2], val[3], val[4], val[5]);
    }

    input.close();
}

void TransferFunction::onOpacityTFChanged()
{
    //std::cout<<"Opacity changed"<<std::endl;
    if(compositeTex)
    {
        checkCudaErrors(cudaDestroyTextureObject(compositeTex));
        compositeTex = 0;
    }

    opacityTF->GetTable(0.0, 1.0, TABLE_SIZE, opacityTable);
    size_t j = 3;
    maxOpacity = -1.f;
    for(size_t i = 0; i < TABLE_SIZE; ++i)
    {
        maxOpacity = fmaxf(maxOpacity, opacityTable[i]);
        compositeTable[j] = opacityTable[i];
        j += 4;
    }

    checkCudaErrors(cudaMemcpyToArray(array, 0, 0, compositeTable, sizeof(float) * TABLE_SIZE * 4, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaCreateTextureObject(&compositeTex, &resourceDesc, &texDesc, NULL));

    //signal changed
    Changed();
}

void TransferFunction::onColorTFChanged()
{
    //std::cout<<"Color changed"<<std::endl;
    if(compositeTex)
    {
        checkCudaErrors(cudaDestroyTextureObject(compositeTex));
        compositeTex = 0;
    }

    colorTF->GetTable(0.0, 1.0, TABLE_SIZE, colorTable);
    size_t j = 0, k = 0;
    for(size_t i = 0; i < TABLE_SIZE; ++i)
    {
        compositeTable[j++] = colorTable[k++];
        compositeTable[j++] = colorTable[k++];
        compositeTable[j++] = colorTable[k++];
        j++;
    }

    checkCudaErrors(cudaMemcpyToArray(array, 0, 0, compositeTable, sizeof(float) * TABLE_SIZE * 4, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaCreateTextureObject(&compositeTex, &resourceDesc, &texDesc, NULL));

    //signal changed
    Changed();
}

void TransferFunction::SetTFConfiguration(uint32_t n, float *index, float *rgb, float *alpha)
{
    opacityTF->RemoveAllPoints();
    colorTF->RemoveAllPoints();

    for(auto i = 0; i < n; ++i)
    {
        opacityTF->AddPoint(index[i], alpha[i]);
        colorTF->AddRGBPoint(index[i], rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
    }

    if(compositeTex)
    {
        checkCudaErrors(cudaDestroyTextureObject(compositeTex));
        compositeTex = 0;
    }

    opacityTF->GetTable(0.0, 1.0, TABLE_SIZE, opacityTable);
    colorTF->GetTable(0.0, 1.0, TABLE_SIZE, colorTable);
    size_t j = 0, k = 0;
    for(size_t i = 0; i < TABLE_SIZE; ++i)
    {
        compositeTable[j++] = colorTable[k++];
        compositeTable[j++] = colorTable[k++];
        compositeTable[j++] = colorTable[k++];
        compositeTable[j++] = opacityTable[i];
    }

    checkCudaErrors(cudaMemcpyToArray(array, 0, 0, compositeTable, sizeof(float) * TABLE_SIZE * 4, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaCreateTextureObject(&compositeTex, &resourceDesc, &texDesc, NULL));
}

cmake_minimum_required(VERSION 3.3)
project(SunVolumeRender)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# CUDA settings
find_package(CUDA QUIET REQUIRED)
list(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -use_fast_math -Xptxas -v -maxrregcount=32)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
set(CUDA_INCLUDE_DIRS /usr/local/cuda/include/)
include_directories(${CUDA_INCLUDE_DIRS})

#Qt
find_package(Qt4 COMPONENTS QtCore QtGui QtOpenGL REQUIRED)
include(${QT_USE_FILE})
set(QT_USE_QTOPENGL TRUE)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(RESOURCES_FILES qdarkstyle/style.qrc resources/images.qrc)

#VTK
#set(VTK_DIR /Volumes/Free/SDK/CTK/build/VTK-build)
set(VTK_DIR /Users/sunwj/Desktop/SDK/CTK/build/VTK-build)
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

#CTK
#set(CTK_DIR /Volumes/Free/SDK/CTK/build/)
set(CTK_DIR /Users/sunwj/Desktop/SDK/CTK/build)
find_package(CTK REQUIRED)
include(${CTK_USE_FILE})
set(CTK_LIBRARIES CTKCore CTKWidgets CTKVisualizationVTKCore CTKVisualizationVTKWidgets)

#GLM
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

set(HOST_SOURCES main.cpp
        gui/mainwindow.cpp
        gui/canvas.cpp
        gui/transferfunction.cpp
        gui/qcustomplot.cpp
        core/VolumeReader.cpp
        core/lights/lights.cpp gui/AddAreaLightDialog.cpp gui/AddAreaLightDialog.h)

set(DEVICE_SOURCES pathtracer.cu)

cuda_compile(DEVICE_OBJS ${DEVICE_SOURCES})

add_executable(SunVolumeRender ${HOST_SOURCES} ${RESOURCES_FILES} ${DEVICE_OBJS})
target_link_libraries(SunVolumeRender ${QT_LIBRARIES} ${VTK_LIBRARIES} ${CTK_LIBRARIES} ${CUDA_LIBRARIES})

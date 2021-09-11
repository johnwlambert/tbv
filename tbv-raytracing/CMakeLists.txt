# Copyright (c) 2021
# Argo AI, LLC, All Rights Reserved.
# 
# Notice: All information contained herein is, and remains the property
# of Argo AI. The intellectual and technical concepts contained herein
# are proprietary to Argo AI, LLC and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law. This work is licensed under a CC BY-NC-SA 4.0 
# International License.
# 
# Originating Authors: John Lambert


cmake_minimum_required(VERSION 3.5)
find_package(CUDA REQUIRED)
find_package(PythonInterp 3.8.0 REQUIRED)
find_package(PythonLibs 3.8.0 REQUIRED)

find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIRS})

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ./pybind11/include
    )

link_directories(
    /usr/local/cuda/lib64
    )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

cuda_add_library(tbv_raytracing SHARED
  tbv_raytracing.cu)

target_link_libraries(tbv_raytracing
  ${PYTHON_LIBRARIES}
  cudart)

set_target_properties(tbv_raytracing PROPERTIES PREFIX "")

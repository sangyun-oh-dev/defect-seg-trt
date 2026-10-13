cmake_minimum_required(VERSION 3.20)
project(seg_infer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
# RTX 30(86) / 40(89) / 50(120) 시리즈. 본인 GPU에 맞게 -DCMAKE_CUDA_ARCHITECTURES=89 처럼 지정 가능
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 89 120)
endif()

set(TENSORRT_ROOT "" CACHE PATH "TensorRT 설치 폴더 (include/, lib/ 포함)")
option(BUILD_BENCH "OpenCV 기반 벤치마크 실행 파일 빌드" ON)

find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include)
# TensorRT 10.x on Windows: nvinfer_10.lib, Linux: libnvinfer.so
find_library(TENSORRT_NVINFER NAMES nvinfer_10 nvinfer HINTS ${TENSORRT_ROOT} PATH_SUFFIXES lib lib64)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER)
  message(FATAL_ERROR "TensorRT를 찾지 못했습니다. -DTENSORRT_ROOT=<TensorRT 폴더> 를 지정하세요.")
endif()
message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT nvinfer: ${TENSORRT_NVINFER}")

# ---- 추론 DLL ----
add_library(seg_infer SHARED src/seg_infer.cpp src/kernels.cu)
target_include_directories(seg_infer
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${TENSORRT_INCLUDE_DIR})
target_compile_definitions(seg_infer PRIVATE SEG_INFER_EXPORTS)
target_link_libraries(seg_infer PRIVATE CUDA::cudart_static ${TENSORRT_NVINFER})
set_target_properties(seg_infer PROPERTIES CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden)
if(MSVC)
  target_compile_options(seg_infer PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /utf-8>)
endif()

# ---- 벤치마크 (검사기 프로그램 역할) ----
if(BUILD_BENCH)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
  add_executable(seg_bench bench/bench.cpp)
  target_link_libraries(seg_bench PRIVATE seg_infer ${OpenCV_LIBS})
  if(MSVC)
    target_compile_options(seg_bench PRIVATE /W4 /utf-8)
  endif()
endif()

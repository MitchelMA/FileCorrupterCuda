cmake_minimum_required(VERSION 3.26)
project(FileCorrupterCUDA
    DESCRIPTION "FileCorrupter additions consisting of CUDA code"
    VERSION 0.0.1
    LANGUAGES CXX CUDA)

find_package(CUDAToolkit)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20=sm_20 -ccbin=g++13.2)

file(GLOB PUBLIC_EXPORT_FILES src/exports/*h src/exports/*.cpp)
file(GLOB PUBLIC_HEADERS src/include/*.hpp src/include/*.h src/include/*.cuh)
file(GLOB PRIVATE_SOURCES src/source/*.cpp src/source/*.cu)

message("Export file: ${PUBLIC_EXPORT_FILES}")
message("Headers: ${PUBLIC_HEADERS}")
message("Sources: ${PRIVATE_SOURCES}")

add_library(CUDA_LIB STATIC
    ${PUBLIC_HEADERS}
    ${PRIVATE_SOURCES})

target_include_directories(CUDA_LIB PUBLIC src/include)

target_link_options(CUDA_LIB
    PUBLIC
    -lcurand)

set_target_properties(CUDA_LIB
    PROPERTIES
    CUDA_SEPERABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)


add_library(FileCorrupterCUDA SHARED ${PUBLIC_EXPORT_FILES})

target_include_directories(FileCorrupterCUDA PUBLIC src/include)

set_property(TARGET FileCorrupterCUDA
    PROPERTY CUDA_SEPERABLE_COMPILATION ON)

target_link_libraries(FileCorrupterCUDA PUBLIC CUDA_LIB)


cmake_minimum_required(VERSION 3.1.0)
project(homework4 CUDA)
include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
	add_executable(hist_coal hist_coal.cu)
endif ()

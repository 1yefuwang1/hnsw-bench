cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(vectorlite VERSION 0.1.0 LANGUAGES CXX)

SET(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(benchmark CONFIG REQUIRED)
find_package(Annoy CONFIG REQUIRED)

find_package(faiss CONFIG REQUIRED)
find_package(lapack CONFIG REQUIRED)
# find_package(blas CONFIG REQUIRED)

find_path(HNSWLIB_INCLUDE_DIRS hnswlib/hnswlib.h)
message(STATUS "HNSWLIB include dir: ${HNSWLIB_INCLUDE_DIRS}")


add_executable(bench bench.cpp)
target_include_directories(bench PUBLIC ${PROJECT_BINARY_DIR} ${HNSWLIB_INCLUDE_DIRS})
target_link_libraries(bench PRIVATE Annoy::Annoy benchmark::benchmark benchmark::benchmark_main faiss lapack)
target_compile_definitions(bench PRIVATE USE_AVX)
# target_compile_options(bench PRIVATE -march=native)
target_compile_options(bench PRIVATE -mavx2)
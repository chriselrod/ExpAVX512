cmake_minimum_required(VERSION 3.20)

project(
  EXPAVX512Benchmarks
  VERSION 1.0
  LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory("${PROJECT_SOURCE_DIR}/.." "extern_build/expavx512")

file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(${PROJECT_NAME} ${sources})
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)
target_compile_options(${PROJECT_NAME} PRIVATE -save-temps -Wall -Wextra -masm=intel)
# package manager doesn't build with LIBPFM ON 
include(FetchContent)
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG 144d23cf5fa0b1b9dd138bc56601920d83e350c7
  OVERRIDE_FIND_PACKAGE)
FetchContent_MakeAvailable()
find_package(benchmark REQUIRED)

set(BENCHMARK_ENABLE_TESTING
    OFF
    CACHE INTERNAL "Disable testing")
set(BENCHMARK_ENABLE_LIBPFM
    ON
    CACHE INTERNAL "Enable Perf Counters")

target_link_libraries(${PROJECT_NAME} EXPAVX512 benchmark::benchmark_main)


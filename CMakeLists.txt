cmake_minimum_required(VERSION 3.20)

project(
  EXPAVX512
  VERSION 1.0
  LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(${PROJECT_NAME} INTERFACE include/ExpAVX512.hpp)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)

target_include_directories(
  ${PROJECT_NAME}
  INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include/${PROJECT_NAME}-${PROJECT_VERSION}>)

# I figure eroring may be better than producing binaries that may not run
# -march=x86-64-v4 or -march=skylake-avx512
target_compile_options(${PROJECT_NAME} INTERFACE -march=native)


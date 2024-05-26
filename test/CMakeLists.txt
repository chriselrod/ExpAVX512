cmake_minimum_required(VERSION 3.20)

project(
  EXPAVX512Test
  VERSION 1.0
  LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory("${PROJECT_SOURCE_DIR}/.." "extern_build/expavx512")

file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(${PROJECT_NAME} ${sources})
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)

find_package(GTest REQUIRED)

target_link_libraries(${PROJECT_NAME} EXPAVX512 GTest::gtest_main)

enable_testing()

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})

cmake_minimum_required(VERSION 3.20)

project(
  odss
  VERSION 0.1.0
  DESCRIPTION "High-performance orbital-debris simulation framework"
  LANGUAGES CXX
)

include(CTest)

option(ODSS_BUILD_PYTHON "Build the Python extension module" ON)
set(ODSS_SANITIZER "none" CACHE STRING "Sanitizer: none, address, or undefined")
set_property(CACHE ODSS_SANITIZER PROPERTY STRINGS none address undefined)

add_library(odss_warnings INTERFACE)
add_library(odss::warnings ALIAS odss_warnings)
target_compile_options(
  odss_warnings
  INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

add_library(odss_sanitizer INTERFACE)
add_library(odss::sanitizer ALIAS odss_sanitizer)
if(ODSS_SANITIZER STREQUAL "address")
  target_compile_options(odss_sanitizer INTERFACE -fsanitize=address -fno-omit-frame-pointer)
  target_link_options(odss_sanitizer INTERFACE -fsanitize=address)
elseif(ODSS_SANITIZER STREQUAL "undefined")
  target_compile_options(odss_sanitizer INTERFACE -fsanitize=undefined -fno-omit-frame-pointer)
  target_link_options(odss_sanitizer INTERFACE -fsanitize=undefined)
elseif(NOT ODSS_SANITIZER STREQUAL "none")
  message(FATAL_ERROR "ODSS_SANITIZER must be none, address, or undefined")
endif()

add_library(odss_core STATIC cpp/src/version.cpp)
add_library(odss::core ALIAS odss_core)
set_target_properties(odss_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(odss_core PUBLIC cxx_std_20)
target_include_directories(
  odss_core
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(
  odss_core
  PRIVATE odss::warnings
  PUBLIC odss::sanitizer
)

if(ODSS_BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core MODULE bindings/python/module.cpp)
  target_compile_features(_core PRIVATE cxx_std_20)
  target_link_libraries(_core PRIVATE odss::core odss::warnings)
  install(TARGETS _core LIBRARY DESTINATION odss)
endif()

if(BUILD_TESTING)
  find_package(GTest CONFIG REQUIRED)
  add_executable(test_version tests/cpp/test_version.cpp)
  target_compile_features(test_version PRIVATE cxx_std_20)
  target_link_libraries(
    test_version
    PRIVATE odss::core odss::warnings GTest::gtest_main
  )
  include(GoogleTest)
  gtest_discover_tests(test_version)
endif()

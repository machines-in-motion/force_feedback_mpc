cmake_minimum_required(VERSION 3.22)

# Project properties
set(PROJECT_NAMESPACE machines-in-motion)
set(PROJECT_NAME force_feedback_mpc)
set(PROJECT_DESCRIPTION "Optimal control tools to achieve force feedback in MPC.")
set(PROJECT_URL https://github.com/${PROJECT_NAMESPACE}/${PROJECT_NAME})

set(CMAKE_VERBOSE_MAKEFILE on)

# Project options
option(SUFFIX_SO_VERSION "Suffix library name with its version" ON)
option(CHECK_RUNTIME_MALLOC "Check eigen mallocs" OFF)
option(BUILD_BENCHMARKS "Build the benchmarks" OFF)

# Project configuration
if(NOT BUILD_STANDALONE_PYTHON_INTERFACE)
  set(PROJECT_USE_CMAKE_EXPORT TRUE)
endif()

# Check if the submodule cmake have been initialized
if(EXISTS "${CMAKE_SOURCE_DIR}/cmake/base.cmake")
  message(STATUS "JRL cmakemodules found in the source tree.")
  set(JRL_CMAKE_MODULES "${CMAKE_SOURCE_DIR}/cmake")
else()
  find_package(jrl-cmakemodules QUIET CONFIG)
  if(jrl-cmakemodules_FOUND)
    get_property(
      JRL_CMAKE_MODULES
      TARGET jrl-cmakemodules::jrl-cmakemodules
      PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "JRL cmakemodules found on system at ${JRL_CMAKE_MODULES}")
  else()
    message(STATUS "JRL cmakemodules not found. Let's fetch it.")
    include(FetchContent)
    FetchContent_Declare(
      "jrl-cmakemodules"
      GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git")
    FetchContent_MakeAvailable("jrl-cmakemodules")
    FetchContent_GetProperties("jrl-cmakemodules" SOURCE_DIR JRL_CMAKE_MODULES)
  endif()
endif()

# JRL-cmakemodule setup
include("${JRL_CMAKE_MODULES}/base.cmake")
include("${JRL_CMAKE_MODULES}/boost.cmake")
include("${JRL_CMAKE_MODULES}/python.cmake")
include("${JRL_CMAKE_MODULES}/apple.cmake")

# Project definition
COMPUTE_PROJECT_ARGS(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

add_compile_options(-Wno-error)
apply_default_apple_configuration()

# Project dependencies
add_project_dependency(OpenMP REQUIRED COMPONENTS CXX)
add_project_dependency(crocoddyl REQUIRED)
add_project_dependency(pinocchio REQUIRED)
if(BUILD_PYTHON_INTERFACE)
  add_project_dependency(eigenpy 3.0.0 REQUIRED)
  FINDPYTHON()
  string(REGEX REPLACE "-" "_" PY_NAME ${PROJECT_NAME})
  set(${PY_NAME}_INSTALL_DIR ${PYTHON_SITELIB}/${PY_NAME})
endif()

if(CHECK_RUNTIME_MALLOC)
  message(STATUS "Check if Eigen memory allocations are performed at runtime.")
  add_definitions(-DFORCE_FEEDBACK_MPC_EIGEN_CHECK_MALLOC)
  add_definitions(-DEIGEN_RUNTIME_NO_MALLOC)
  message(STATUS "Add debug flags and no optimization. Valgrind can be used.")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0 -fstack-protector")
endif()

# Main Library
if(BUILD_STANDALONE_PYTHON_INTERFACE)
  add_project_dependency(${PROJECT_NAME} REQUIRED CONFIG)
else()
  set(${PROJECT_NAME}_HEADERS
    include/${PROJECT_NAME}/lowpassfilter/state.hpp
    include/${PROJECT_NAME}/lowpassfilter/action.hpp
    include/${PROJECT_NAME}/softcontact/state.hpp
    include/${PROJECT_NAME}/softcontact/dam-augmented.hpp
    include/${PROJECT_NAME}/softcontact/dam3d-augmented.hpp
    include/${PROJECT_NAME}/softcontact/dam1d-augmented.hpp
    include/${PROJECT_NAME}/softcontact/iam-augmented.hpp
    # include/${PROJECT_NAME}/softcontact/multicontact.hpp
    include/${PROJECT_NAME}/frictioncone/residual-friction-cone.hpp
    include/${PROJECT_NAME}/frictioncone/residual-friction-cone-augmented.hpp
  )
  set(${PROJECT_NAME}_SOURCES
    src/lowpassfilter/state.cpp
    src/lowpassfilter/action.cpp
    src/softcontact/state.cpp
    src/softcontact/dam-augmented.cpp
    src/softcontact/dam3d-augmented.cpp
    src/softcontact/dam1d-augmented.cpp
    src/softcontact/iam-augmented.cpp
    # src/softcontact/multicontact.cpp
    src/frictioncone/residual-friction-cone.cpp
    src/frictioncone/residual-friction-cone-augmented.cpp
  )

  add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})
  add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
  target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include>)
  if(SUFFIX_SO_VERSION)
    set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION ${PROJECT_VERSION})
  endif()
  target_link_libraries(${PROJECT_NAME} crocoddyl::crocoddyl)
  set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/../lib")
  set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
endif()

# Python Bindings
if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(bindings)
  add_subdirectory(python)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# Unit tests
if(BUILD_TESTING AND NOT BUILD_STANDALONE_PYTHON_INTERFACE)
  add_project_dependency(Boost REQUIRED COMPONENTS unit_test_framework)
  add_project_dependency(example-robot-data 4.0.9 REQUIRED PKG_CONFIG_REQUIRES
"example-robot-data >= 4.0.9")
  add_subdirectory(tests)
endif()
if(BUILD_TESTING AND BUILD_PYTHON_INTERFACE)
  add_subdirectory(tests/python)
endif()

# Installation
if(NOT BUILD_STANDALONE_PYTHON_INTERFACE)
  install(TARGETS ${PROJECT_NAME} EXPORT ${TARGETS_EXPORT_NAME} DESTINATION lib)
  install(FILES package.xml DESTINATION share/${PROJECT_NAME})
endif()

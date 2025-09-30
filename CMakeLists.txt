cmake_minimum_required(VERSION 3.10)

# Project properties
set(PROJECT_NAMESPACE machines-in-motion)
set(PROJECT_NAME force_feedback_mpc)
set(PROJECT_DESCRIPTION "Optimal control tools to achieve force feedback in MPC.")
set(PROJECT_URL https://github.com/${PROJECT_NAMESPACE}/${PROJECT_NAME})

set(CMAKE_VERBOSE_MAKEFILE on)

# Project options
option(BUILD_PYTHON_INTERFACE "Build the python binding" ON)
option(SUFFIX_SO_VERSION "Suffix library name with its version" ON)
option(CHECK_RUNTIME_MALLOC "Check eigen mallocs" OFF)
option(BUILD_BENCHMARKS "Build the benchmarks" OFF)

# Project configuration
set(PROJECT_USE_CMAKE_EXPORT TRUE)
set(CUSTOM_HEADER_DIR ${PROJECT_NAME})

# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
IF(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/base.cmake")
  IF(${CMAKE_VERSION} VERSION_LESS "3.14.0")
    MESSAGE(FATAL_ERROR "\nPlease run the following command first:\ngit submodule update --init\n")
  ELSE()
    include(FetchContent)
    FetchContent_Declare("jrl-cmakemodules"
      GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git")
    FetchContent_MakeAvailable("jrl-cmakemodules")
    FetchContent_GetProperties("jrl-cmakemodules" SOURCE_DIR JRL_CMAKE_MODULES)
  ENDIF()
ENDIF()

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
find_package(Boost REQUIRED COMPONENTS unit_test_framework)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_project_dependency(crocoddyl REQUIRED)
add_project_dependency(pinocchio REQUIRED)
add_project_dependency(example-robot-data 4.0.7 REQUIRED PKG_CONFIG_REQUIRES
"example-robot-data >= 4.0.7")

if(BUILD_PYTHON_INTERFACE)
  add_project_dependency(eigenpy 3.0.0 REQUIRED)
  FINDPYTHON()
  string(REGEX REPLACE "-" "_" PY_NAME ${PROJECT_NAME})
  set(${PY_NAME}_INSTALL_DIR ${PYTHON_SITELIB}/${PY_NAME})
  SEARCH_FOR_BOOST_PYTHON()
endif()

if(CHECK_RUNTIME_MALLOC)
  message(STATUS "Check if some memory allocations are performed at runtime.")
  add_definitions(-DFORCE_FEEDBACK_MPC_EIGEN_CHECK_MALLOC)
  add_definitions(-DEIGEN_RUNTIME_NO_MALLOC)
endif()

# Main Library
set(${PROJECT_NAME}_HEADERS
  include/${PROJECT_NAME}/python.hpp
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
target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include>)

if(SUFFIX_SO_VERSION)
  set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION ${PROJECT_VERSION})
endif()

# Main Executable
TARGET_LINK_LIBRARIES(${PROJECT_NAME} crocoddyl::crocoddyl)
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/../lib")
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# Python Bindings
if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(bindings)
endif()

# # Benchmarks
# if(BUILD_BENCHMARKS)
#   add_subdirectory(benchmarks)
# endif()

# # Unit tests
# add_subdirectory(tests)

# Python packages 
add_subdirectory(python)

# Installation
install(TARGETS ${PROJECT_NAME} EXPORT ${TARGETS_EXPORT_NAME} DESTINATION lib)
install(FILES package.xml DESTINATION share/${PROJECT_NAME})

# # Add flags for Valgrind (if desired)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fstack-protector")  # Add debug symbols and no optimization

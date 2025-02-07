cmake_minimum_required(VERSION 3.10)
project(pose4dof VERSION 0.1.0 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wno-sign-compare -Wno-unused-parameter)

option(POSE4DOF_ENABLE_TESTS "Build unit tets" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_EXECUTABLES "Build executable files for test" OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include_directories(include)

find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)

###########################################################################
add_library(${PROJECT_NAME} SHARED
  src/Pose4DoF.cpp
  include/pose4dof/Pose4DoF.h )

target_include_directories(${PROJECT_NAME} PUBLIC
  ${GTSAM_INCLUDE_DIR}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  # INSTALL
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    gtsam
    Eigen3::Eigen
)

set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
add_library(pose4dof::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

###########################################################################
# Define executables
if (BUILD_EXECUTABLES)
  add_executable(pose2_numerical_derivative
  src/pose2_numerical_derivative.cpp
  src/Pose4DoF.cpp)
  target_link_libraries(pose2_numerical_derivative gtsam Eigen3::Eigen)

  add_executable(Pose4DoFSLAMExample src/Pose4DoFSLAMExample.cpp src/Pose4DoF.cpp)
  target_link_libraries(Pose4DoFSLAMExample gtsam Eigen3::Eigen)

  add_executable(Pose3SLAMExample src/Pose3SLAMExample.cpp src/Pose4DoF.cpp)
  target_link_libraries(Pose3SLAMExample gtsam Eigen3::Eigen)
endif()

###########################################################################
# Tests
if(POSE4DOF_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

###########################################################################
# Installation (from pose_graph_tools)
install(
  TARGETS ${PROJECT_NAME}
  EXPORT pose4dof-targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
  EXPORT pose4dof-targets
  FILE pose4dofTargets.cmake
  NAMESPACE pose4dof::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pose4dof)

configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/pose4dofConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/pose4dofConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pose4dof)
write_basic_package_version_file(
  pose4dofConfigVersion.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pose4dofConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/pose4dofConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pose4dof)

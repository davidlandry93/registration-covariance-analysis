cmake_minimum_required(VERSION 3.7)
project(recova VERSION 0.1)

option(PYTHON_BINDINGS "Build the python bindings of the c++ code" ON)

find_package(Eigen3 REQUIRED)
find_package(gflags REQUIRED)
find_package(lieroy REQUIRED)
find_package(libnabo REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(nabo_adapter src/nabo_adapter.cpp)
target_include_directories(nabo_adapter
  PUBLIC include ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(nabo_adapter nabo ${OpenMP_CXX_FLAGS})
target_compile_options(nabo_adapter PRIVATE ${OpenMP_CXX_FLAGS})
set_property(TARGET nabo_adapter PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(recova
  src/util.cpp
  src/centered_clustering.cpp
  src/pointcloud_separator.cpp
  src/grid_pointcloud_separator.cpp
  src/pointcloud_logger_locator.cpp
  src/pointcloud_logger.cpp
  src/xyz_pointcloud_logger.cpp)
target_include_directories(recova
  PUBLIC include ${EIGEN3_INCLUDE_DIRS})
set_property(TARGET recova PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(centered_clustering src/centered_clustering_main.cpp src/centered_clustering.cpp src/greedy_seed_selection_algorithm.cpp src/localized_seed_selection_algorithm.cpp)
target_include_directories(centered_clustering PRIVATE include)
target_link_libraries(centered_clustering recova nabo_adapter gflags lieroy::lieroy)

add_executable(grid_pointcloud_separator src/grid_pointcloud_separator_main.cpp)
target_link_libraries(grid_pointcloud_separator recova gflags)

add_executable(point_density src/point_density_main.cpp src/point_density.cpp)
target_include_directories(point_density PRIVATE include)
target_link_libraries(point_density recova nabo_adapter gflags)

add_executable(overlapping_region src/overlapping_region_main.cpp)
target_link_libraries(overlapping_region gflags nabo_adapter recova)
target_include_directories(overlapping_region PRIVATE include)

add_executable(json2xyz src/json2xyz_main.cpp)
target_include_directories(json2xyz PRIVATE include)


# Installation
install(TARGETS
  centered_clustering
  grid_pointcloud_separator
  json2xyz
  overlapping_region
  point_density
  RUNTIME DESTINATION bin)

install(PROGRAMS bin/xyz2vtk bin/all_xyz2vtk bin/recov_explore_alpha DESTINATION bin)

if(PYTHON_BINDINGS)
  add_subdirectory(recova_core)
endif(PYTHON_BINDINGS)

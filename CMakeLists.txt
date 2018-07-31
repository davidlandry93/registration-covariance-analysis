cmake_minimum_required(VERSION 3.5)
project(recova VERSION 0.1)

option(PYTHON_BINDINGS "Build the python bindings of the c++ code" ON)

find_package(lieroy REQUIRED)
find_package(Boost 1.58 EXACT)
find_package(Eigen3 REQUIRED)
find_package(gflags REQUIRED)
find_package(glog REQUIRED)
find_package(libnabo REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PCL 1.3 REQUIRED COMPONENTS common io features)
find_package(recov REQUIRED)

add_definitions(${PCL_DEFINITIONS})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(nabo_adapter src/nabo_adapter.cpp)
target_include_directories(nabo_adapter
  PUBLIC include ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(nabo_adapter ${libnabo_LIBRARIES} ${OpenMP_CXX_FLAGS})
target_compile_options(nabo_adapter PRIVATE ${OpenMP_CXX_FLAGS})
set_property(TARGET nabo_adapter PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(recova
  src/util.cpp
  src/centered_clustering.cpp
  src/localized_seed_selection_algorithm.cpp
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



message(STATUS ${PCL_INCLUDE_DIRS})
message(STATUS ${EIGEN3_INCLUDE_DIRS})

include_directories(${PCL_INCLUDE_DIRS})

add_executable(normals_of_cloud src/normals_of_cloud_main.cpp)
target_include_directories(normals_of_cloud PRIVATE include ${EIGEN3_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} PUBLIC)
target_link_libraries(normals_of_cloud recov::recov gflags ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_FEATURES_LIBRARIES})

# Installation
install(TARGETS
  centered_clustering
  grid_pointcloud_separator
  json2xyz
  overlapping_region
  point_density
  normals_of_cloud
  RUNTIME DESTINATION bin)

install(PROGRAMS bin/xyz2vtk bin/all_xyz2vtk bin/recov_explore_alpha bin/one_registration DESTINATION bin)

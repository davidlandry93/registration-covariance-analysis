cmake_minimum_required(VERSION 3.0)
project(recova VERSION 0.1)

find_package(Eigen3 REQUIRED)
find_package(gflags REQUIRED)
find_package(libnabo REQUIRED)
find_package(OpenMP REQUIRED)

add_library(nabo_adapter src/nabo_adapter.cpp)
target_include_directories(nabo_adapter
  PUBLIC ${EIGEN3_INCLUDE_DIRS}
  PRIVATE include)
target_link_libraries(nabo_adapter nabo ${OpenMP_CXX_FLAGS})
target_compile_options(nabo_adapter PRIVATE ${OpenMP_CXX_FLAGS})
set_property(TARGET nabo_adapter PROPERTY CXX_STANDARD 11)

add_executable(centered_clustering src/centered_clustering_main.cpp src/centered_clustering.cpp)
target_include_directories(centered_clustering PRIVATE include)
target_link_libraries(centered_clustering nabo_adapter gflags)
set_property(TARGET centered_clustering PROPERTY CXX_STANDARD 11)


# Installation
install(TARGETS centered_clustering
  RUNTIME DESTINATION bin)

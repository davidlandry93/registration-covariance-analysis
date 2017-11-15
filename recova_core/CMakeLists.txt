cmake_minimum_required(VERSION 3.7)

# Python bindings.

option(USE_PYTHON3 "Use Python3 instead of Python2" ON)
option(AUTO_INSTALL_BINDINGS "Install the python bindings every time they are built." OFF)

if(USE_PYTHON3)
  set(Python_ADDITIONAL_VERSION 3)
  find_package(Boost 1.63 COMPONENTS python3 numpy3 REQUIRED)
else(USE_PYTHON3)
  find_package(Boost 1.63 COMPONENTS python numpy REQUIRED)
endif(USE_PYTHON3)

find_package(PythonLibs)


add_library(recova_bindings SHARED recova_core.cpp)
target_link_libraries(recova_bindings recova nabo_adapter ${Boost_LIBRARIES})
target_include_directories(recova_bindings PRIVATE ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
set_target_properties(recova_bindings PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY recova_core
  PREFIX ""
  OUTPUT_NAME "recova_core")
set_property(TARGET recova_bindings PROPERTY CXX_STANDARD 11)

add_custom_command(TARGET recova_bindings POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_CURRENT_SOURCE_DIR}/recova_core $<TARGET_FILE_DIR:recova_bindings>)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/setup.py)



# Python bindings installation.

find_program(PIP "pip")

if(PIP)
  set(PYTHON_BINDINGS_INSTALLATION_COMMAND ${PIP} install --force-reinstall .)

  # Install the python package at install time.
  install(CODE "execute_process(COMMAND ${PYTHON_BINDINGS_INSTALLATION_COMMAND} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")

  if(AUTO_INSTALL_BINDINGS)
    add_custom_command(TARGET recova_bindings POST_BUILD
      COMMAND ${PYTHON_BINDINGS_INSTALLATION_COMMAND}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif(AUTO_INSTALL_BINDINGS)
endif(PIP)

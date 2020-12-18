# BSD 3-clause
# Copyright (c) 2020, New York University and Max Planck Gesellschaft
cmake_minimum_required(VERSION 3.5.1)

project(blmc_controllers)

# required to use std::shared_ptr in code and to link the python bindings
if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed")
endif()
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -W -Wextra -fPIC")

# ensuring path to libraries are set during install
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

############################
# Search for dependencies. #
############################

#
# Dependencies
#
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(eiquadprog REQUIRED)

#local depends
find_package(mpi_cmake_modules REQUIRED)
find_package(yaml_utils REQUIRED)

###########################
# Create the main library #
###########################

add_library(${PROJECT_NAME}
    src/impedance_controller.cpp
    src/centroidal_pd_controller.cpp
    src/centroidal_force_qp_controller.cpp
)
target_link_libraries(${PROJECT_NAME} pinocchio::pinocchio)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} eiquadprog::eiquadprog)
target_link_libraries(${PROJECT_NAME} yaml_utils::yaml_utils)

# Includes. Add the include dependencies
target_include_directories(
  ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include>)

################################
# Install the package          #
################################

# command to install the library and binaries
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include
)

# we also need to install the header files
install(DIRECTORY include/ DESTINATION include)

# ################################
# # python bindings with pybind11#
# ################################
pybind11_add_module(py_${PROJECT_NAME} MODULE
    srcpy/${PROJECT_NAME}.cpp
    srcpy/impedance_controller.cpp
    srcpy/centroidal_pd_controller.cpp
    srcpy/centroidal_force_qp_controller.cpp
)


target_link_libraries(py_${PROJECT_NAME} PRIVATE pybind11::module)
target_link_libraries(py_${PROJECT_NAME} PRIVATE ${PROJECT_NAME})

# # install the bindings
get_python_install_dir(PYTHON_INSTALL_DIR)
install(TARGETS py_${PROJECT_NAME} DESTINATION ${PYTHON_INSTALL_DIR})

################################
# install python packages      #
################################

# install the python package too
install(
    DIRECTORY python/${PROJECT_NAME}
    DESTINATION "${PYTHON_INSTALL_DIR}"
    PATTERN "*.pyc" EXCLUDE
    PATTERN "__pycache__" EXCLUDE
  )

##########################
# building documentation #
##########################
add_documentation()

############################
# create the cmake package #
############################
generate_cmake_package(INSTALL_EXPORT)

cmake_minimum_required(VERSION 3.18)
project(geon_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(GEON_EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen")
set(GEON_NANOFLANN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/nanoflann/include")

# use local pybind11 or fall back to system install
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11/CMakeLists.txt")
  add_subdirectory(external/pybind11)
else()
  find_package(pybind11 CONFIG REQUIRED)
endif()

pybind11_add_module(
  features
  MODULE
  native/features/python_wrapper.cpp
  native/features/features.cpp
)
target_include_directories(features PRIVATE "${GEON_EIGEN_DIR}")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(features PRIVATE OpenMP::OpenMP_CXX)
endif()

set(GEON_NATIVE_INSTALL_DIR "geon/_native")

install(
  TARGETS features
  LIBRARY DESTINATION ${GEON_NATIVE_INSTALL_DIR}
  RUNTIME DESTINATION ${GEON_NATIVE_INSTALL_DIR}
  ARCHIVE DESTINATION ${GEON_NATIVE_INSTALL_DIR}
)

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/geon/"
  DESTINATION geon
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "*.pyi"
    PATTERN "__pycache__" EXCLUDE
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/resources/" DESTINATION geon/resources)

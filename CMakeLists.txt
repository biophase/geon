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

pybind11_add_module(
  region_growing
  MODULE
  native/region_growing/python_wrapper.cpp
  native/region_growing/rgrow.cpp
  native/region_growing/connected_components.cpp
)
target_include_directories(
  region_growing
  PRIVATE
    "${GEON_EIGEN_DIR}"
    "${GEON_NANOFLANN_DIR}"
)

pybind11_add_module(
  plane_ransac
  MODULE
  native/region_growing/plane_ransac_python_wrapper.cpp
  native/region_growing/plane_ransac.cpp
)
target_include_directories(
  plane_ransac
  PRIVATE
    "${GEON_EIGEN_DIR}"
    "${GEON_NANOFLANN_DIR}"
)

pybind11_add_module(
  superpoints
  MODULE
  native/region_growing/superpoints_python_wrapper.cpp
  native/region_growing/superpoints.cpp
  native/region_growing/cp_d0_dist.cpp
  native/region_growing/cut_pursuit.cpp
  native/region_growing/cut_pursuit_d0.cpp
  native/region_growing/maxflow.cpp
)
target_include_directories(
  superpoints
  PRIVATE
    "${GEON_EIGEN_DIR}"
    "${GEON_NANOFLANN_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/native/region_growing"
)
target_compile_definitions(superpoints PRIVATE COMP_T_ON_32_BITS)

pybind11_add_module(
  region_merge
  MODULE
  native/region_growing/region_merge_python_wrapper.cpp
  native/region_growing/region_merge.cpp
)
target_include_directories(
  region_merge
  PRIVATE
    "${GEON_EIGEN_DIR}"
    "${GEON_NANOFLANN_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/native/region_growing"
)

pybind11_add_module(
  subsampling
  MODULE
  native/subsampling_python_wrapper.cpp
  native/subsampling.cpp
)
target_include_directories(subsampling PRIVATE "${GEON_EIGEN_DIR}")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(features PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(plane_ransac PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(superpoints PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(region_merge PRIVATE OpenMP::OpenMP_CXX)
endif()

set(GEON_NATIVE_INSTALL_DIR "geon/_native")

install(
  TARGETS features region_growing plane_ransac superpoints region_merge subsampling
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

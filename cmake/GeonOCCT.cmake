# OCCT assumes it owns CMAKE_SOURCE_DIR, so configure it as an isolated project.
include(ExternalProject)
set(GEON_OCCT_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/external/opencascade")
if(NOT EXISTS "${GEON_OCCT_SOURCE}/src/GeomAPI/GeomAPI_IntSS.hxx")
  message(FATAL_ERROR "OCCT source missing. Run git submodule update --init --recursive")
endif()
set(GEON_OCCT_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/occt-install")
set(GEON_OCCT_LIBRARIES TKGeomAlgo TKBRep TKGeomBase TKG3d TKG2d TKMath TKernel)
set(GEON_OCCT_BYPRODUCTS)
foreach(toolkit IN LISTS GEON_OCCT_LIBRARIES)
  list(APPEND GEON_OCCT_BYPRODUCTS "${GEON_OCCT_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${toolkit}${CMAKE_STATIC_LIBRARY_SUFFIX}")
endforeach()
file(MAKE_DIRECTORY "${GEON_OCCT_PREFIX}/include/opencascade")
ExternalProject_Add(geon_occt
  SOURCE_DIR "${GEON_OCCT_SOURCE}"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/occt-build"
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden
    -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
    -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL
    -DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON
    -DCMAKE_EXPORT_PACKAGE_REGISTRY=OFF
    -DBUILD_LIBRARY_TYPE=Static
    -DBUILD_CPP_STANDARD=C++17
    -DBUILD_MODULE_FoundationClasses=OFF
    -DBUILD_MODULE_ModelingData=OFF
    -DBUILD_MODULE_ModelingAlgorithms=OFF
    -DBUILD_MODULE_Visualization=OFF
    -DBUILD_MODULE_ApplicationFramework=OFF
    -DBUILD_MODULE_DataExchange=OFF
    -DBUILD_MODULE_DETools=OFF
    -DBUILD_MODULE_Draw=OFF
    -DBUILD_ADDITIONAL_TOOLKITS=TKGeomAlgo
    -DBUILD_DOC_Overview=OFF
    -DUSE_TK=OFF -DUSE_FREETYPE=OFF -DUSE_OPENGL=OFF -DUSE_XLIB=OFF
    -DUSE_TBB=OFF -DUSE_VTK=OFF -DUSE_RAPIDJSON=OFF
    -DINSTALL_DIR=${GEON_OCCT_PREFIX}
    -DINSTALL_DIR_LAYOUT=Unix -DINSTALL_DIR_LIB=lib
    -DINSTALL_DIR_INCLUDE=include/opencascade
  BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release --parallel 4
  INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR> --config Release
  BUILD_BYPRODUCTS ${GEON_OCCT_BYPRODUCTS}
  LOG_CONFIGURE ON LOG_BUILD ON LOG_INSTALL ON
)
foreach(toolkit IN LISTS GEON_OCCT_LIBRARIES)
  add_library(geon_occt_${toolkit} STATIC IMPORTED)
  set_target_properties(geon_occt_${toolkit} PROPERTIES
    IMPORTED_LOCATION "${GEON_OCCT_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${toolkit}${CMAKE_STATIC_LIBRARY_SUFFIX}"
    INTERFACE_INCLUDE_DIRECTORIES "${GEON_OCCT_PREFIX}/include/opencascade")
  add_dependencies(geon_occt_${toolkit} geon_occt)
endforeach()
pybind11_add_module(surface_geometry MODULE native/surface_geometry.cpp)
add_dependencies(surface_geometry geon_occt)
target_compile_definitions(surface_geometry PRIVATE OCCT_STATIC_BUILD)
foreach(toolkit IN LISTS GEON_OCCT_LIBRARIES)
  target_link_libraries(surface_geometry PRIVATE geon_occt_${toolkit})
endforeach()
if(WIN32)
  target_link_libraries(surface_geometry PRIVATE advapi32 user32 gdi32 ws2_32 psapi)
else()
  find_package(Threads REQUIRED)
  target_link_libraries(surface_geometry PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
  if(UNIX AND NOT APPLE)
    target_link_options(surface_geometry PRIVATE "LINKER:--exclude-libs,ALL")
  endif()
endif()
install(TARGETS surface_geometry LIBRARY DESTINATION geon/_native RUNTIME DESTINATION geon/_native)
install(FILES "${GEON_OCCT_SOURCE}/LICENSE_LGPL_21.txt" "${GEON_OCCT_SOURCE}/OCCT_LGPL_EXCEPTION.txt"
  DESTINATION geon/resources/licenses/opencascade)

cmake_minimum_required(VERSION 3.23)
project(gaussianfilter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")

set(CMAKE_VERBOSE_MAKEFILE True)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file" -G Ninja)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/bin.dbg)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin.rel)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/bin.relwithdbg)

# Set the vcpkg triplet
set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "Vcpkg target triplet")

find_package(OpenCV REQUIRED COMPONENTS core highgui imgcodecs imgproc videoio)
include_directories(${OpenCV_INCLUDE_DIRS})

if(WIN32)
    set(OpenCV_RUNTIME_PATH "${OpenCV_DIR}/../../bin")
else()
    set(OpenCV_RUNTIME_PATH "${OpenCV_DIR}/../../lib")
endif()

add_subdirectory(filter)
add_subdirectory(prj.test)
add_subdirectory(ctikz)

find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc)
    add_custom_target(doxygen ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)

add_custom_target(build_all
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
        DEPENDS filter prj.test ctikz
        COMMENT "Building all targets..."
        )

install(TARGETS filter DESTINATION bin)
install(DIRECTORY filter/include/ DESTINATION include/filter)
install(FILES filter/src/filter.cpp DESTINATION filter/src)

install(DIRECTORY prj.test/ DESTINATION bin/prj.test)
install(DIRECTORY cmake/ DESTINATION .)

install(TARGETS ctikz DESTINATION bin)
install(DIRECTORY ctikz/include/ DESTINATION include/ctikz)
install(FILES ctikz/ctikz.cpp DESTINATION ctikz)
install(FILES ctikz/include/ctikz/ctikz.hpp DESTINATION include/ctikz)

# Install OpenCV runtime files
if(NOT WIN32)
    set(LIB_SUFFIX ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
else()
    set(LIB_SUFFIX ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

install(DIRECTORY "${LIB_SUFFIX}/opencv" DESTINATION bin)

install(FILES CMakeLists.txt filter/CMakeLists.txt prj.test/CMakeLists.txt ctikz/CMakeLists.txt DESTINATION .)

       









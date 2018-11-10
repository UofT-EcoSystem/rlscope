# NOTE: make sure to run ./setup.sh before attempting to build this.
# setup.sh will download the tensorflow C-API.
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(DNN_Tensorflow_CPP LANGUAGES CXX)

set(IGNORE_ERRORS "-Wno-unused")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Werror ${IGNORE_ERRORS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -demangle")

if(MINGW OR CYGWIN)
    set(IS_WINDOWS ON)
else()
    set(IS_WINDOWS OFF)
endif()

if(IS_WINDOWS)
    message("MINGW or CYGWIN is set")
endif()

if (NOT IS_WINDOWS AND NOT EXISTS ${PROJECT_SOURCE_DIR}/external_libs/lib)
    message(FATAL_ERROR "You need to run setup.sh first.")
endif()

#if (NOT EXISTS ./external_libs/lib)
#   message(FATAL_ERROR "You need to run setup.sh first.")
#endif()

#link_directories(
#        ${PROJECT_SOURCE_DIR}/external_libs/lib
#)
include_directories(
        ./
        ./src
        ./external_libs/include
)

if(NOT IS_WINDOWS)
    find_package( Boost 1.58 REQUIRED )
    include_directories( ${Boost_INCLUDE_DIR} )
#    add_executable( anyExecutable myMain.cpp )
endif()
#target_link_libraries( anyExecutable LINK_PUBLIC ${Boost_LIBRARIES} )

set(SOURCES)
# ( cd ~/clone/dnn_tensorflow_cpp; find -type f | grep --perl-regexp '\.(c|cc|cpp|cxx)$' | xargs -n1 dirname | sort --unique | grep -v CMake; ) | cboard
set(SRC_DIRS
        ./src
        ./src/dqn
        ./src/simulator
        ./tensorflow/c
        ./tensorflow/core/platform
        ./tensorflow/core/platform/default
        ./tensorflow/core/platform/posix
        )

foreach(direc ${SRC_DIRS})

    file(GLOB direc_sources ${direc}/*.c)
    list(APPEND SOURCES ${direc_sources})

    file(GLOB direc_sources ${direc}/*.cc)
    list(APPEND SOURCES ${direc_sources})

    file(GLOB direc_sources ${direc}/*.cpp)
    list(APPEND SOURCES ${direc_sources})

    file(GLOB direc_sources ${direc}/*.cxx)
    list(APPEND SOURCES ${direc_sources})

endforeach()

message("Sources = ${SOURCES}")

#add_executable(dummy ${SOURCES})

if(IS_WINDOWS)
    # Include copies of headers for libraries that are too annoying to get on windows.
    include_directories(
            windows_includes/
    )
endif()


add_executable(model ${SOURCES})
if(NOT IS_WINDOWS)
#    find_library(TENSORFLOW_LIB tensorflow
#            REQUIRED
#            HINTS "${PROJECT_SOURCE_DIR}/external_libs/lib")
    find_library(TENSORFLOW_LIB tensorflow_framework
            REQUIRED
            HINTS "${PROJECT_SOURCE_DIR}/external_libs/lib")
    message("> TENSORFLOW_LIB = ${TENSORFLOW_LIB}")
    target_link_libraries(model ${TENSORFLOW_LIB})
endif()

if(NOT IS_WINDOWS)
    find_library(GFLAGS_LIB gflags REQUIRED)
    message("> GFLAGS_LIB = ${GFLAGS_LIB}")
    target_link_libraries(model ${GFLAGS_LIB})
endif()

configure_file(normalized_car_features.csv ${CMAKE_CURRENT_BINARY_DIR}/normalized_car_features.csv COPYONLY)
if(MSVC)
    target_compile_definitions(main PRIVATE COMPILER_MSVC)
endif(MSVC)
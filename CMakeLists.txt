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

set(EXEC_NAMES
        tests
        model
        cartpole)
set(GFLAGS_EXEC_NAMES
        model
        cartpole)

set(SOURCES)
# ( cd ~/clone/dnn_tensorflow_cpp; find -type f | grep --perl-regexp '\.(c|cc|cpp|cxx)$' | xargs -n1 dirname | sort --unique | grep -v CMake; ) | cboard

set(SRC_DIRS
        ./src
        # NOTE: don't include drivers.
        # ./src/drivers
        ./src/tf
        ./src/model
        ./src/dqn
        ./src/simulator
        ./tensorflow/c
        ./tensorflow/core/platform
        ./tensorflow/core/platform/default
        ./tensorflow/core/platform/posix
        )

set(TEST_DIRS
        ${SRC_DIRS}
        ./test
        ./test/model
        )

macro(gather_sources srcs_var directories)
    foreach(direc ${${directories}})

        file(GLOB direc_sources ${direc}/*.c)
        list(APPEND ${srcs_var} ${direc_sources})

        file(GLOB direc_sources ${direc}/*.cc)
        list(APPEND ${srcs_var} ${direc_sources})

        file(GLOB direc_sources ${direc}/*.cpp)
        list(APPEND ${srcs_var} ${direc_sources})

        file(GLOB direc_sources ${direc}/*.cxx)
        list(APPEND ${srcs_var} ${direc_sources})

    endforeach()
endmacro()

gather_sources(SOURCES SRC_DIRS)
gather_sources(TEST_SOURCES TEST_DIRS)

message("Sources = ${SOURCES}")
message("Test sources = ${TEST_SOURCES}")

#add_executable(dummy ${SOURCES})

if(IS_WINDOWS)
    # Include copies of headers for libraries that are too annoying to get on windows.
    include_directories(
            windows_includes/
    )
endif()

# BLAZ
add_executable(model
        ./src/drivers/run_model.cpp
        ${SOURCES})
add_executable(cartpole
        ./src/drivers/run_cartpole.cpp
        ${SOURCES})
add_executable(tests
        ./src/drivers/run_tests.cpp
        ${TEST_SOURCES})

if(NOT IS_WINDOWS)
    find_package( Boost 1.58 REQUIRED
            COMPONENTS system filesystem)
    include_directories( ${Boost_INCLUDE_DIR} )
    foreach(executable ${EXEC_NAMES})
        target_link_libraries(${executable}
                ${Boost_LIBRARIES}
                #                ${Boost_FILESYSTEM_LIBRARY}
                #                ${Boost_SYSTEM_LIBRARY}
                )
    endforeach()
endif()

if(NOT IS_WINDOWS)
#    find_library(TENSORFLOW_LIB tensorflow
#            REQUIRED
#            HINTS "${PROJECT_SOURCE_DIR}/external_libs/lib")
    find_library(TENSORFLOW_LIB tensorflow_framework
            REQUIRED
            HINTS "${PROJECT_SOURCE_DIR}/external_libs/lib")
    message("> TENSORFLOW_LIB = ${TENSORFLOW_LIB}")
    foreach(executable ${EXEC_NAMES})
        target_link_libraries(${executable} ${TENSORFLOW_LIB})
    endforeach()
endif()

if(NOT IS_WINDOWS)
    find_library(GFLAGS_LIB gflags REQUIRED)
    message("> GFLAGS_LIB = ${GFLAGS_LIB}")
    foreach(executable ${GFLAGS_EXEC_NAMES})
        target_link_libraries(${executable} ${GFLAGS_LIB})
    endforeach()
endif()

if(NOT IS_WINDOWS)
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    add_subdirectory(third_party/json)
endif()

#if(NOT IS_WINDOWS)
#    find_library(GTEST_LIB gtest REQUIRED)
#    message("> GTEST_LIB = ${GTEST_LIB}")
#    target_link_libraries(tests ${GTEST_LIB})
#endif()

if(NOT IS_WINDOWS)
    find_package(GTest REQUIRED)
    include_directories(${GTEST_INCLUDE_DIRS})
    target_link_libraries(tests ${GTEST_LIBRARY})
    foreach(executable ${EXEC_NAMES})
        target_link_libraries(${executable} pthread)
        target_link_libraries(${executable} nlohmann_json::nlohmann_json)
    endforeach()
endif()

configure_file(normalized_car_features.csv ${CMAKE_CURRENT_BINARY_DIR}/normalized_car_features.csv COPYONLY)
if(MSVC)
    target_compile_definitions(main PRIVATE COMPILER_MSVC)
endif(MSVC)
# In CLion, add -DCLION=on to enable development on windows machine (so CMakeLists.txt loads properly).
#
# NOTE: make sure to run ./setup.sh before attempting to build this.
# setup.sh will download the tensorflow C-API.
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
#https://github.com/robertmaynard/code-samples/blob/master/posts/cmake/CMakeLists.txt

if(MINGW OR CYGWIN OR WIN32 OR MSVC OR CLION)
#    message(FATAL_ERROR "WHY")
    set(IS_WINDOWS ON)
else()
    set(IS_WINDOWS OFF)
endif()

if(IS_WINDOWS)
    message("> Detected Windows development machine")
endif()

if(IS_WINDOWS)
    set(LANGUAGES CXX)
else()
    set(LANGUAGES CXX CUDA)
endif()
project(DNN_Tensorflow_CPP LANGUAGES ${LANGUAGES})

set(IGNORE_ERRORS "-Wno-unused")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Werror ${IGNORE_ERRORS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -demangle")

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
set(ALL_TARGETS
        ${EXEC_NAMES}
        ${GFLAGS_EXEC_NAMES}
        py_interface
        )

set(SOURCES)
# ( cd ~/clone/dnn_tensorflow_cpp; find -type f | grep --perl-regexp '\.(c|cc|cpp|cxx)$' | xargs -n1 dirname | sort --unique | grep -v CMake; ) | cboard

set(TF_SRC_DIRS
        ./tensorflow/c
        ./tensorflow/core/platform
        ./tensorflow/core/platform/default
        ./tensorflow/core/platform/posix
        )

set(SRC_DIRS
        ./src
        # NOTE: don't include drivers.
        # ./src/drivers
        ./src/tf
        ./src/model
        ./src/dqn
        ./src/simulator
        ${TF_SRC_DIRS}
        )
set(PY_INTERFACE_LIB_DIRS
        ./test/py_interface)

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

        file(GLOB direc_sources ${direc}/*.cu)
        list(APPEND ${srcs_var} ${direc_sources})

    endforeach()
endmacro()

gather_sources(SOURCES SRC_DIRS)
gather_sources(TEST_SOURCES TEST_DIRS)
gather_sources(TF_SOURCES TF_SRC_DIRS)
gather_sources(PY_INTERFACE_LIB_SOURCES PY_INTERFACE_LIB_DIRS)

if(IS_WINDOWS)
    set(CUPTI_DIRS
            ./CUPTI/include
            ./CUPTI/include/GL
            ./CUPTI/include/openacc
            ./CUPTI/sample
            ./CUPTI/sample/activity_trace_async
            ./CUPTI/sample/callback_event
            ./CUPTI/sample/callback_metric
            ./CUPTI/sample/callback_timestamp
            ./CUPTI/sample/cupti_query
            ./CUPTI/sample/event_multi_gpu
            ./CUPTI/sample/event_sampling
            ./CUPTI/sample/nvlink_bandwidth
            ./CUPTI/sample/openacc_trace
            ./CUPTI/sample/pc_sampling
            ./CUPTI/sample/sass_source_map
            ./CUPTI/sample/unified_memory
            )
#    message("CUPTI_DIRS = ${CUPTI_DIRS}")
#    message("CUPTI_SOURCES = ${CUPTI_SOURCES}")
    gather_sources(CUPTI_SOURCES CUPTI_DIRS)
    add_executable(cupti_dummy ${CUPTI_SOURCES})
endif()

message("Sources = ${SOURCES}")
message("Test sources = ${TEST_SOURCES}")

#add_executable(dummy ${SOURCES})

if(IS_WINDOWS)
    # Include copies of headers for libraries that are too annoying to get on windows.
    include_directories(
            windows_includes/
            CUPTI/include
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

#./test/py_interface/py_interface_src.cpp
add_library(py_interface SHARED
        ${PY_INTERFACE_LIB_SOURCES}
#        ${TF_SOURCES}
        )

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
    message("> TENSORFLOW_LIB HINT = ${PROJECT_SOURCE_DIR}/external_libs/lib")
    find_library(TENSORFLOW_LIB tensorflow
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
    message("CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}")
    find_package(GTest REQUIRED)
    # find_package(gtest REQUIRED)
    include_directories(${GTEST_INCLUDE_DIRS})
    target_link_libraries(tests ${GTEST_LIBRARY})
    foreach(executable ${EXEC_NAMES})
        target_link_libraries(${executable} pthread)
        target_link_libraries(${executable} nlohmann_json::nlohmann_json)
    endforeach()
endif()

foreach(trg ${ALL_TARGETS})
    target_compile_features(${trg} PUBLIC cxx_std_11)
endforeach()

configure_file(normalized_car_features.csv ${CMAKE_CURRENT_BINARY_DIR}/normalized_car_features.csv COPYONLY)
if(MSVC)
    target_compile_definitions(main PRIVATE COMPILER_MSVC)
endif(MSVC)
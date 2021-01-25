# In CLion, add -DCLION=on to enable development on windows machine (so CMakeLists.txt loads properly).
#
# NOTE: make sure to run ./setup.sh before attempting to build this.
# setup.sh will download the tensorflow C-API.

# You can use a locally built protobuf (3.9.1), but you will need to provide
# a path to where you install protobuf's include directory:
#
# $ mkdir build
# $ cd build
# $ cmake -DProtobuf_INCLUDE_DIR=$HOME/protobuf/include -DProtobuf_LIBRARY=$HOME/protobuf/lib/libprotobuf.so ..

cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
set(LANGUAGES CXX CUDA)
project(rlscope LANGUAGES ${LANGUAGES})

#
# Enable/disable building different libraries/executables.
#
option(RLS_ENABLE_TESTS
        "Build rls-test program for running C++ unit-tests that cover offline analysis component of RL-Scope"
        OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
#message("> CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}")

# What to use for:
# target_link_libraries(${TARGET} PRIVATE ...)
set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)

include(LibFindMacros)

if(CLION)
    SET(ENV{LD_LIBRARY_PATH} "${CMAKE_SOURCE_DIR}/local.host/${CMAKE_BUILD_TYPE}/lib:$ENV{LD_LIBRARY_PATH}")
    message("Set LD_LIBRARY_PATH = $ENV{LD_LIBRARY_PATH}")

    SET(ENV{PATH} "${CMAKE_SOURCE_DIR}/local.host/${CMAKE_BUILD_TYPE}/bin:$ENV{PATH}")
    message("Set PATH = $ENV{PATH}")

    SET(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/local.host/${CMAKE_BUILD_TYPE}")
    message("Set CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
endif()

if(DEFINED ENV{RLSCOPE_INSTALL_PREFIX})
    message("> Building for docker:")
    set(RLSCOPE_INSTALL_PREFIX $ENV{RLSCOPE_INSTALL_PREFIX})
else()
    message("> Building for host")
    set(RLSCOPE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/local.host/${CMAKE_BUILD_TYPE}")
endif()
message("  RLSCOPE_INSTALL_PREFIX=${RLSCOPE_INSTALL_PREFIX}")
# Override system installation of libraries (e.g., gflags)
message("    link_directories: ${RLSCOPE_INSTALL_PREFIX}/${CMAKE_BUILD_TYPE}/lib")
link_directories(BEFORE ${RLSCOPE_INSTALL_PREFIX}/${CMAKE_BUILD_TYPE}/lib)
message("    include_directories: ${RLSCOPE_INSTALL_PREFIX}/${CMAKE_BUILD_TYPE}/include")
include_directories(BEFORE ${RLSCOPE_INSTALL_PREFIX}/${CMAKE_BUILD_TYPE}/include)


list(PREPEND CMAKE_PROGRAM_PATH "${RLSCOPE_INSTALL_PREFIX}/${CMAKE_BUILD_TYPE}/bin")
list(PREPEND CMAKE_INCLUDE_PATH "${RLSCOPE_INSTALL_PREFIX}/${CMAKE_BUILD_TYPE}/include")
list(PREPEND CMAKE_INCLUDE_PATH "${RLSCOPE_INSTALL_PREFIX}/${CMAKE_BUILD_TYPE}/bin")
list(PREPEND CMAKE_LIBRARY_PATH "${RLSCOPE_INSTALL_PREFIX}/${CMAKE_BUILD_TYPE}/lib")

# Use pkg-config to get hints about paths
#libfind_pkg_check_modules(CUDAProfilingAPI_PKGCONF CUDAProfilingAPI)

find_package(CUDAProfilingAPI REQUIRED)
#TRY_ALIAS_LIB(cuda::nvperf_host nvperf_host_LIBRARY)
#TRY_ALIAS_LIB(cuda::nvperf_target nvperf_target_LIBRARY)
#TRY_ALIAS_LIB(cuda::nvtx nvtx_LIBRARY)
#TRY_ALIAS_LIB(cuda::cupti cupti_LIBRARY)

#include_directories(${CUDAProfilingAPI_INCLUDE_DIR})
# Gets included by libraries that require it (e.g., profilerhost_util)
#link_libraries(${CUDAProfilingAPI_LIBRARY})

# https://stackoverflow.com/questions/32469953/why-is-cmake-designed-so-that-it-removes-runtime-path-when-installing
#list(APPEND CMAKE_INSTALL_RPATH ${BOOST_LIBRARYDIR})
# NOTE: This isn't enough to keep BOOST_LIBRARYDIR in RUNPATH for some reason.
# See CMAKE_INSTALL_RPATH related line below.

# Keep CUDA/CUPTI paths in the RUNPATH of binaries and libraries (i.e., librlscope.so, rls-analyze).
# This allows libcupti and libcuda to be located on typical Linux distributions, without the user
# needing to modify LD_LIBRARY_PATH manually.
# But of course, they can always override it by setting LD_LIBRARY_PATH if they so choose.
#
# This option is best described here:
# https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling#what-is-rpath
#   Describes the order of searching for libraries on UNIX (first LD_LIBRARY_PATH, then RUNPATH).
# https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling#always-full-rpath
#   Describes how CMake handles library paths in general.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# We want to be able to do std::move() for lambda captures (c++11 doesn't have that).
#set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(DEBUG_CMAKE_VERBOSE OFF)

set(ENABLE_BACKTRACE ON CACHE BOOL "Enable pretty-printed stacktrace during LOG(FATAL) or DCHECK failures; requires adding -g debug symbols")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Not until cmake 3.17
set(CUDA_PATH /usr/local/cuda)
if (NOT EXISTS ${CUDA_PATH} AND IS_DIRECTORY ${CUDA_PATH})
    message(FATAL_ERROR
            "You must set CUDA_PATH to the root of your CUDA installation (didn't find it at ${CUDA_PATH})")
endif()

#set(WITH_CUDA_LD_PRELOAD ON CACHE BOOL "Wrap CUDA API calls using LD_PRELOAD trick; do this to see whether wrapping CUDA API calls manually is faster than using libcupti.")
##set(WITH_CUDA_LD_PRELOAD OFF CACHE BOOL "Wrap CUDA API calls using LD_PRELOAD trick; do this to see whether wrapping CUDA API calls manually is faster than using libcupti.")

#set(WITH_CUDA_LD_PRELOAD ON BOOL "Wrap CUDA API calls using LD_PRELOAD trick; do this to see whether wrapping CUDA API calls manually is faster than using libcupti.")
#set(WITH_CUDA_LD_PRELOAD OFF BOOL "Wrap CUDA API calls using LD_PRELOAD trick; do this to see whether wrapping CUDA API calls manually is faster than using libcupti.")
#set(WITH_CUDA_LD_PRELOAD OFF BOOL)
#set(WITH_CUDA_LD_PRELOAD OFF)

# NOTE: don't use a cached cmake variable here, since if we switch the cache variable on/off from this file it WON'T affect
# the current built (since it's persisted...i.e. the whole purpose of cache variables).
#set(WITH_CUDA_LD_PRELOAD ON)
set(WITH_CUDA_LD_PRELOAD OFF)
# Implement database queries and construction of numpy data-formats in C++ instead of python
# PRO: speedup data preparation
# CON: extra dependency (libpqxx)
#set(WITH_CXX_SQL OFF)
# Implement reading raw trace-files into eo_times format using C++.
set(WITH_CXX_ANALYSIS ON)
#message("> WITH_CUDA_LD_PRELOAD = ${WITH_CUDA_LD_PRELOAD}")

set(BOOST_VERSION 1.73.0)
set(BOOST_VERSION_UNDERSCORES 1_73_0)

#set(PROTOBUF_VERSION 3.6.1)
set(PROTOBUF_VERSION 3.9.1)


#
# If windows, don't run rest of cmake file.
#
if(MINGW OR CYGWIN OR WIN32 OR MSVC)
    #    message(FATAL_ERROR "WHY")
    message("> Detected Windows development machine")
    set(IS_WINDOWS ON)
else()
    message("> Detected Unix setup (build environment)")
    set(IS_WINDOWS OFF)
endif()
if(IS_WINDOWS)
    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/CMakeLists.windows.txt")
        message(FATAL_ERROR "ERROR: expected to find CMakeLists.windows.txt containing files we should index in CLion")
    endif()
    message("> Loading CMakeLists.windows.txt (windows dev environment)")
    include(CMakeLists.windows.txt)
    message("> Exit CMakeLists.txt early (windows dev environment)")
    return()
endif()

#
# Check that user ran "bash setup.sh" to download ./third_party libraries
#
set(THIRD_PARTY_LIB_DIRNAMES
        json
        abseil-cpp
        nsync
        backward-cpp
        eigen
        # NOTE: protobuf dependency is optional.
        # If setup.sh installs it, we will use it.
        # This allows us to build protobuf ahead of time during the container build.
        # protobuf-${PROTOBUF_VERSION}
        boost_${BOOST_VERSION_UNDERSCORES}
        spdlog
        CTPL
        googletest
        gflags
        )
macro(CheckSetupScriptRan VAR ERR_MSG)
    set(${VAR} TRUE)
    foreach(THIRD_PARTY_LIB_DIRNAME IN LISTS THIRD_PARTY_LIB_DIRNAMES)
        set(THIRD_PARTY_LIB_PATH ${PROJECT_SOURCE_DIR}/third_party/${THIRD_PARTY_LIB_DIRNAME})
        if (NOT EXISTS ${THIRD_PARTY_LIB_PATH})
            set(${ERR_MSG} "couldn't find third_party library at ${THIRD_PARTY_LIB_PATH}")
            set(${VAR} FALSE)
            break()
        endif()
    endforeach()
endmacro()
CheckSetupScriptRan(HAS_SETUP_RAN ERR_MSG)
if (NOT HAS_SETUP_RAN)
    message(FATAL_ERROR
            "ERROR: ${ERR_MSG}\n"
            "You need to run setup.sh first:\n"
            "  $ cd ${PROJECT_SOURCE_DIR}\n"
            "  $ bash ./setup.sh")
endif()

#
# Run find_package for ./third_party libraries
#

# Third party: protobuf
set(Protobuf_USE_STATIC_LIBS ON)
find_package(Protobuf REQUIRED)
#message("> Protobuf_INCLUDE_DIRS = ${Protobuf_INCLUDE_DIRS}")
#message("> Protobuf_LIBRARIES = ${Protobuf_LIBRARIES}")
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Third party: double_conversion
# https://cmake.org/cmake/help/v3.9/module/FindProtobuf.html
include(double_conversion)
find_package(DoubleConversion REQUIRED)
#message("> double_conversion_INCLUDE_DIRS = ${double_conversion_INCLUDE_DIRS}")
#message("> double_conversion_LIBRARIES = ${double_conversion_LIBRARIES}")

# Third party: nsync
add_library(nsync_includes INTERFACE)
target_include_directories(nsync_includes INTERFACE third_party/nsync/public)
set(NSYNC_ENABLE_TESTS OFF)
#set(BUILD_SHARED_LIBS ON)
add_subdirectory(third_party/nsync
        # DON'T include nsync's install(...) files in CPack package (e.g., include/nsync*.h)
        EXCLUDE_FROM_ALL)

# Third party: abseil-cpp
# https://github.com/abseil/abseil-cpp/blob/master/CMake/README.md
set(ABSL_RUN_TESTS OFF)
set(BUILD_TESTING OFF)
add_subdirectory(third_party/abseil-cpp
        # DON'T include abseil-cpp's install(...) files in CPack package (e.g., include/nsync*.h)
        EXCLUDE_FROM_ALL)
unset(BUILD_TESTING)

# Third party: backward-cpp
IF(CMAKE_BUILD_TYPE STREQUAL "Release")
    message("> Disable C++ stacktraces for ${CMAKE_BUILD_TYPE}")
    set(STACK_DETAILS_AUTO_DETECT FALSE)
ENDIF()
add_subdirectory(third_party/backward-cpp
        # DON'T include backward-cpp's install(...) files in CPack package (e.g., include/nsync*.h)
        EXCLUDE_FROM_ALL)

# Third party: eigen
find_package(Eigen3 REQUIRED)

# Third party: json
#set(JSON_BuildTests OFF CACHE INTERNAL "")
#add_subdirectory(third_party/json)
find_package(nlohmann_json REQUIRED)
#...
#add_library(foo ...)
#...

#
# Compiler definitions.
#

# NOTE: -Wno-deprecated-declarations
#   Get rid of "‘template<class> class std::auto_ptr’ is deprecated" errors
#   when compiling gpu_util_experiment.
#   NOTE: attempts to set target specific compile options didn't work since
#   nvcc ignored them... not sure why.
set(IGNORE_ERRORS "-Wno-unused-result -Wno-unused -Wno-unknown-pragmas -Wno-deprecated-declarations")
set(ADD_WARNINGS_FLAGS "-Wuninitialized")
#set(CMAKE_CXX_STANDARD 11)
#-std=c++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ADD_WARNINGS_FLAGS} -Wall -Werror ${IGNORE_ERRORS} ")
#-Wl,--unresolved-symbols=ignore-in-shared-libs
#-Wl,--unresolved-symbols=report-all
set(RLSCOPE_LINKER_FLAGS "-demangle -Wl,--no-undefined")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${RLSCOPE_LINKER_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${RLSCOPE_LINKER_FLAGS}")

# Address sanitizer doesn't appear to work on CUDA programs...
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
#set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

include_directories(
        ./
        ./src
        ./src/libs/include
        ./src/libs
        ./external_libs/include
)

set(ANALYSIS_SRC_DIRS
        ./src/analysis
        )

set(COMMON_SRC_DIRS
        ./src/common
        )

set(SRC_DIRS
        # ./src
        ./src/analysis
        ./src/common
        ./src/cuda_api_profiler
        ./src/drivers
        ./src/experiment

        # ./src/drivers
        # ./src/tf
        # ./src/model
        # ./src/dqn
        # ./src/simulator
        )

#
# Gather paths of source files for creating RLScope targets (i.e., binaries, libraries)
#

macro(RmGTestSources SRCS_VAR)
    list(FILTER ${SRCS_VAR} EXCLUDE REGEX "/test/")
endmacro()

macro(_glob_sources glob_type srcs_var direc)
    set(direc_sources)

    file(${glob_type} direc_sources ${direc}/*.c)
    list(APPEND ${srcs_var} ${direc_sources})

    file(${glob_type} direc_sources ${direc}/*.cc)
    list(APPEND ${srcs_var} ${direc_sources})

    file(${glob_type} direc_sources ${direc}/*.cpp)
    list(APPEND ${srcs_var} ${direc_sources})

    file(${glob_type} direc_sources ${direc}/*.cxx)
    list(APPEND ${srcs_var} ${direc_sources})

    file(${glob_type} direc_sources ${direc}/*.cu)
    list(APPEND ${srcs_var} ${direc_sources})
endmacro()

macro(_glob_gtest_sources glob_type srcs_var direc)
    set(direc_sources)

    file(${glob_type} direc_sources ${direc}/test_*.c)
    list(APPEND ${srcs_var} ${direc_sources})

    file(${glob_type} direc_sources ${direc}/test_*.cc)
    list(APPEND ${srcs_var} ${direc_sources})

    file(${glob_type} direc_sources ${direc}/test_*.cpp)
    list(APPEND ${srcs_var} ${direc_sources})

    file(${glob_type} direc_sources ${direc}/test_*.cxx)
    list(APPEND ${srcs_var} ${direc_sources})

    file(${glob_type} direc_sources ${direc}/test_*.cu)
    list(APPEND ${srcs_var} ${direc_sources})
endmacro()

macro(gather_sources srcs_var directories)
    foreach(direc ${${directories}})
        _glob_sources(GLOB ${srcs_var} ${direc})
    endforeach()
endmacro()

macro(gather_sources_recursive srcs_var directories)
    foreach(direc ${${directories}})
        _glob_sources(GLOB_RECURSE ${srcs_var} ${direc})
    endforeach()
endmacro()

macro(gather_gtest_sources srcs_var directories)
    foreach(direc ${${directories}})
        _glob_gtest_sources(GLOB_RECURSE ${srcs_var} ${direc})
    endforeach()
endmacro()

# Find directories named "test" with ./src
# These are gtest unit tests for libraries in ./src/libs
set(SOURCE_ROOTS
        ./src
        ./test)
gather_gtest_sources(RLS_TEST_SRCS SOURCE_ROOTS)

gather_sources(ANALYSIS_SOURCES ANALYSIS_SRC_DIRS)
RmGTestSources(ANALYSIS_SOURCES)
gather_sources(COMMON_SOURCES COMMON_SRC_DIRS)
RmGTestSources(COMMON_SOURCES)

set(CUDA_PROF_DIRS
        ./src/cuda_api_profiler
        )
include_directories(/usr/local/cuda/extras/CUPTI/include)
include_directories(/usr/local/cuda/include)

gather_sources(CUDA_PROF_SRCS CUDA_PROF_DIRS)
RmGTestSources(CUDA_PROF_SRCS)

# https://github.com/bombela/backward-cpp/tree/84ae4f5e80381aca765a0810d4c811acae3cd7c7#configuration--dependencies

if (DEBUG_CMAKE_VERBOSE)
    message("> CUDA_PROF_SRCS = ${CUDA_PROF_SRCS}")
    message("> ANALYSIS_SOURCES = ${ANALYSIS_SOURCES}")
    message("> COMMON_SOURCES = ${COMMON_SOURCES}")
    #message("> TEST_ANALYSIS_SOURCES = ${TEST_ANALYSIS_SOURCES}")
    message("> RLSCOPE_PROTO_SRCS = ${RLSCOPE_PROTO_SRCS}")
    message("> RLSCOPE_PROTO_HDRS = ${RLSCOPE_PROTO_HDRS}")
    message("> RLS_TEST_SRCS = ${RLS_TEST_SRCS}")
    message("> BACKWARD_ENABLE = ${BACKWARD_ENABLE}")
endif()


# With PROTOBUF_GENERATE_CPP_APPEND_PATH=OFF
#     In CMake 3.5.1 (Ubuntu 16.04) this makes the
#     generated protobuf .pb.cc and .pb.h maintain their original directory structure:
#       rlscope/build/tensorflow/core/lib/core/error_codes.pb.h
#     Without this, cmake's FindProtobuf module will still all generated files in the rlscope/build directory:
#       rlscope/build/error_codes.pb.h
# HOWEVER; this no longer works in CMake 3.14.5 (Ubuntu 18.04)
#       The directory structure for generated files becomes:
#         rlscope/build/CMakeFiles/tensorflow/core/lib/core/error_codes.pb.h
#       BUT, RLSCOPE_PROTO_SRCS still matches this directory structure (non-existent files):
#         rlscope/build/error_codes.pb.h
# TLDR: Oh well, just generate cpp files in rlscope/build/*.pb.cpp
# set(PROTOBUF_GENERATE_CPP_APPEND_PATH OFF)
protobuf_generate_cpp(RLSCOPE_PROTO_SRCS RLSCOPE_PROTO_HDRS
        rlscope/protobuf/rlscope_prof.proto
        rlscope/protobuf/pyprof.proto
        )
set(SAMPLE_CUDA_API_SOURCES

        # Add SIGSEGV handler that prints out stack-trace.
        # NOTE: For some reason this doesn't always work and gdb is required to debug...
        # I think that's because the program being executed overrides the SIGSEGV handler
        # (which is FINE, as long as backward-cpp doesn't override the application handler).
        ${BACKWARD_ENABLE}
        # NOTE: we should probably put "common" stuff between 'analysis' and libsample in the common directory,
        # instead of pulling in all of the analysis sources.
        ${COMMON_SOURCES}
        ${RLSCOPE_PROTO_SRCS} ${RLSCOPE_PROTO_HDRS}
        ${CUDA_PROF_SRCS}
        )
if (NOT WITH_CUDA_LD_PRELOAD)
    message("> WITH_CUDA_LD_PRELOAD not set; don't compile cuda_ld_preload.cc")

    list(FILTER SAMPLE_CUDA_API_SOURCES EXCLUDE REGEX "cuda_api_profiler/cuda_ld_preload.cc")
    list(FILTER SAMPLE_CUDA_API_SOURCES EXCLUDE REGEX "cuda_api_profiler/cuda_ld_preload.h")
endif()
if (DEBUG_CMAKE_VERBOSE)
    message("> SAMPLE_CUDA_API_SOURCES: ${SAMPLE_CUDA_API_SOURCES}")
endif()

if (WITH_CXX_ANALYSIS)
    set(RLS_ANALYZE_SRCS
            ./src/drivers/cpp_dump_proto.cpp
            ${ANALYSIS_SOURCES}
            ${COMMON_SOURCES}
            ${BACKWARD_ENABLE}
            ${RLSCOPE_PROTO_SRCS} ${RLSCOPE_PROTO_HDRS} src/analysis/lru_cache.cc src/analysis/sample_periods.cc src/analysis/sample_periods.h)
    list(FILTER RLS_ANALYZE_SRCS EXCLUDE REGEX "/c_test_util.cc")
    if (DEBUG_CMAKE_VERBOSE)
        message("> RLS_ANALYZE_SRCS = ${RLS_ANALYZE_SRCS}")
    endif()

endif()

#
# Functions for creating targets.
#

function(AddFlags TARGET PROPERTY NEW_FLAGS)
    #https://stackoverflow.com/questions/36662920/xcode-clang-link-build-dynamic-framework-or-dylib-not-embed-dependencies
    #    set(TARGET "sample_cuda_api")
    #    set(NEW_FLAGS "-Wl,-undefined,dynamic_lookup")
    get_target_property(OLD_FLAGS ${TARGET} ${PROPERTY})
    if(OLD_FLAGS STREQUAL "OLD_FLAGS-NOTFOUND")
        SET(OLD_FLAGS "") # Set to empty string
    else()
        SET(OLD_FLAGS "${OLD_FLAGS} ") # A space to cleanly separate from existing content
    endif()
    # Append our values
    SET(OLD_FLAGS "${OLD_FLAGS}${NEW_FLAGS}" )
    set_target_properties(${TARGET} PROPERTIES ${PROPERTY} "${OLD_FLAGS}" )
endfunction()

function(AddCUDA TARGET)
    find_package(CUDA REQUIRED)
    # message(">> CUDA_LIBRARIES = ${CUDA_LIBRARIES}")
    target_link_libraries(${TARGET} PRIVATE ${CUDA_LIBRARIES})
    target_include_directories(${TARGET} PRIVATE ${CUDA_INCLUDE_DIRS})
    # NOTE: We want to link to the CUDA "driver API" which is contained in libcuda.so.
    # Doesn't look like FindCUDA.cmake has a CUDA_*_LIBRARY for it though... oh well.
    target_link_libraries(${TARGET} PRIVATE cuda)
endfunction()

function(AddUnitTest TARGET)
    find_package(GTest REQUIRED)
    # message("> GTEST_LIBRARIES = ${GTEST_LIBRARIES}")
    # message("> GTEST_INCLUDE_DIRS = ${GTEST_INCLUDE_DIRS}")
    target_include_directories(${TARGET} PRIVATE ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${TARGET} PRIVATE ${GTEST_LIBRARIES})
endfunction()

function(AddCUPTIDependencies TARGET)
    find_package(CUDAProfilingAPI REQUIRED)
    target_link_libraries(${TARGET} PUBLIC ${CUDAProfilingAPI_LIBRARIES})
    target_include_directories(${TARGET} PRIVATE ${CUDAProfilingAPI_INCLUDE_DIRS})
    # message("> CUDAProfilingAPI_LIBRARIES = ${CUDAProfilingAPI_LIBRARIES}")
    # message("> CUDAProfilingAPI_INCLUDE_DIRS = ${CUDAProfilingAPI_INCLUDE_DIRS}")
    AddCUDA(${TARGET})
endfunction()

function(FindProtobufDeps)
    # IMPORTANT: link libprotobuf statically into librlscope.so so LD_PRELOAD
    # doesn't override the protobuf library used by libtensorflow_framework.so.
    # librlscope.so is careful to hide its symbols, so libprotobuf.a symbols will be hidden.
    # If we DON'T do this, libprotobuf.so will complain about compile-version and
    # runtime-version discrepancies.
    set(Protobuf_USE_STATIC_LIBS ON)
    find_package(Protobuf REQUIRED)
    include_directories(${Protobuf_INCLUDE_DIRS})
    include_directories(${CMAKE_CURRENT_BINARY_DIR})
endfunction()

function(AddProtobufDeps TARGET)
    FindProtobufDeps()
    target_link_libraries(${TARGET} PRIVATE ${Protobuf_LIBRARIES})
    #message("> Protobuf_INCLUDE_DIRS = ${Protobuf_INCLUDE_DIRS}")
    #message("> Protobuf_LIBRARIES = ${Protobuf_LIBRARIES}")
    target_include_directories(${TARGET} PRIVATE ${Protobuf_INCLUDE_DIRS})
    target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

function(AddTensorRTDeps TARGET)
    find_package(TensorRT REQUIRED)
    #message("> TensorRT_INCLUDE_DIRS = ${TensorRT_INCLUDE_DIRS}")
    #message("> TensorRT_LIBRARIES = ${TensorRT_LIBRARIES}")
    target_link_libraries(${TARGET} PRIVATE ${TensorRT_LIBRARIES})
    target_include_directories(${TARGET} PRIVATE ${TensorRT_INCLUDE_DIRS})
endfunction()

function (AddLoggingDeps TARGET)
    # This works in docker, but no on Ubuntu 18.04 host...
    # so just make our own find_package for spdlog
    # find_package(spdlog REQUIRED)
    # target_link_libraries(${TARGET} PRIVATE spdlog::spdlog)
    find_package(MySpdlog REQUIRED)
    target_link_libraries(${TARGET} PRIVATE ${MySpdlog_LIBRARIES})
    target_include_directories(${TARGET} PRIVATE ${MySpdlog_INCLUDE_DIRS})
    # message("> MySpdlog_LIBRARIES = ${MySpdlog_LIBRARIES}")
    # message("> MySpdlog_INCLUDE_DIRS = ${MySpdlog_INCLUDE_DIRS}")

    IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("> Enable spdlog logging for target=${TARGET}")
        get_target_property(CUR_FLAG ${TARGET} COMPILE_DEFINITIONS)
        if (NOT CUR_FLAG)
            set(CUR_FLAG)
        endif()
        set_target_properties(${TARGET} PROPERTIES COMPILE_DEFINITIONS "${CUR_FLAG};SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG")
        get_target_property(CUR_FLAG ${TARGET} COMPILE_DEFINITIONS)
    ELSE()
        message("> SKIP: spdlog logging for target=${TARGET}")
        get_target_property(CUR_FLAG ${TARGET} COMPILE_DEFINITIONS)
        if (NOT CUR_FLAG)
            set(CUR_FLAG)
        endif()
        set_target_properties(${TARGET} PROPERTIES COMPILE_DEFINITIONS "${CUR_FLAG};SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_OFF")
        get_target_property(CUR_FLAG ${TARGET} COMPILE_DEFINITIONS)
    ENDIF()
endfunction()

function(AddTensorflowDependencies TARGET INCLUDE_CUDA)
    add_dependencies(${TARGET} double_conversion)
    # This will add libraries, definitions and include directories needed by backward
    # by setting each property on the target.
    add_backward(${TARGET})

    if (ENABLE_BACKTRACE)
        set(BACKTRACE_SYMS_FLAGS "-g")
        AddFlags(${TARGET} COMPILE_FLAGS ${BACKTRACE_SYMS_FLAGS})
        AddFlags(${TARGET} LINK_FLAGS ${BACKTRACE_SYMS_FLAGS})
    endif()

    AddProtobufDeps(${TARGET})
#    target_link_libraries(${TARGET} PRIVATE ${Protobuf_LIBRARIES})

    #message("> RLSCOPE_PROTO_SRCS = ${RLSCOPE_PROTO_SRCS}")
    #message("> RLSCOPE_PROTO_HDRS = ${RLSCOPE_PROTO_HDRS}")
    target_link_libraries(${TARGET} PRIVATE
            absl::base
            absl::strings
            absl::memory)
    target_link_libraries(${TARGET} PRIVATE nsync)

    AddLoggingDeps(${TARGET})

    target_link_libraries(${TARGET} PRIVATE nlohmann_json::nlohmann_json)

    # Simple header-only thread pool library.
    set(CTPL_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/third_party/CTPL)
    target_include_directories(${TARGET} PRIVATE ${CTPL_INCLUDE_DIRS})

    # target_link_libraries(${TARGET} PRIVATE boost_filesystem)
    # Statically link third-party libraries so librlscope.so is more easily linked against in C/C++ projects.
    target_link_libraries(${TARGET} PRIVATE libboost_filesystem.a)

    target_link_libraries(${TARGET} PRIVATE nsync_includes)

    target_link_libraries(${TARGET} PRIVATE nsync_cpp)
    target_include_directories(${TARGET} PRIVATE ${double_conversion_INCLUDE_DIR})
    target_link_libraries(${TARGET} PRIVATE ${double_conversion_LIBRARIES})
    target_link_libraries(${TARGET} PRIVATE Eigen3::Eigen)
    target_link_libraries(${TARGET} PRIVATE "-ldl")
    if(INCLUDE_CUDA)
        AddCUPTIDependencies(${TARGET})

        if (NOT WITH_CUDA_LD_PRELOAD)
            AddCUDA(${TARGET})
        endif()

        # Link driver API by adding -lcuda:
        #   https://devtalk.nvidia.com/default/topic/774340/cmake-undefined-reference-cuctxdetach-cumemgetinfo_v2-cuctxcreate_v2-cuinit/
        target_link_libraries(${TARGET} PUBLIC cuda)
        # https://stackoverflow.com/questions/26243169/cmake-target-include-directories-meaning-of-scope
        # NOTE: I still don't fully understand the PUBLIC/PRIVATE/INTERFACE scope here...
        target_include_directories(${TARGET} PRIVATE ${CUDA_INCLUDE_DIRS})

        find_package(Backtrace REQUIRED)
        target_link_libraries(${TARGET} PRIVATE ${Backtrace_LIBRARY})
        target_include_directories(${TARGET} PRIVATE ${Backtrace_INCLUDE_DIR})

    endif()

endfunction()

function(AddRLScopeTarget TARGET)
    add_library(${TARGET} SHARED ${SAMPLE_CUDA_API_SOURCES})
    AddCommonRLSDeps(${TARGET})
    AddLoggingDeps(${TARGET})
#    cuda_add_library(${TARGET} SHARED ${SAMPLE_CUDA_API_SOURCES})
    add_dependencies(${TARGET} double_conversion)
    # This will add libraries, definitions and include directories needed by backward
    # by setting each property on the target.
    add_backward(${TARGET})

    # target_link_libraries(${TARGET} PRIVATE boost_filesystem)
    # Statically link third-party libraries so librlscope.so is more easily linked against in C/C++ projects.
    target_link_libraries(${TARGET} PRIVATE libboost_filesystem.a)

    target_link_libraries(${TARGET} PRIVATE nlohmann_json::nlohmann_json)

#    target_link_libraries(${TARGET} PRIVATE ${Protobuf_LIBRARIES})
    AddProtobufDeps(${TARGET})

    #message("> RLSCOPE_PROTO_SRCS = ${RLSCOPE_PROTO_SRCS}")
    #message("> RLSCOPE_PROTO_HDRS = ${RLSCOPE_PROTO_HDRS}")
    #absl::synchronization
    target_link_libraries(${TARGET} PRIVATE
            absl::base
            absl::strings
            absl::memory)
    target_link_libraries(${TARGET} PRIVATE nsync)
    target_link_libraries(${TARGET} PRIVATE nsync_cpp)
    target_include_directories(${TARGET} PRIVATE ${double_conversion_INCLUDE_DIR})
    target_link_libraries(${TARGET} PRIVATE ${double_conversion_LIBRARIES})
    target_link_libraries(${TARGET} PRIVATE Eigen3::Eigen)
    AddCUPTIDependencies(${TARGET})

    if (NOT WITH_CUDA_LD_PRELOAD)
        AddCUDA(${TARGET})
    endif()

    if (WITH_CUDA_LD_PRELOAD)
        target_link_libraries(${TARGET} PUBLIC "-ldl")
    endif()

    # Link driver API by adding -lcuda:
    #   https://devtalk.nvidia.com/default/topic/774340/cmake-undefined-reference-cuctxdetach-cumemgetinfo_v2-cuctxcreate_v2-cuinit/
    target_link_libraries(${TARGET} PUBLIC cuda)
    # https://stackoverflow.com/questions/26243169/cmake-target-include-directories-meaning-of-scope
    # NOTE: I still don't fully understand the PUBLIC/PRIVATE/INTERFACE scope here...
    target_include_directories(${TARGET} PRIVATE ${CUDA_INCLUDE_DIRS})

    #    add_subdirectory(third_party/boost_${BOOST_VERSION_UNDERSCORES})

    find_package(Backtrace REQUIRED)
    target_link_libraries(${TARGET} PRIVATE ${Backtrace_LIBRARY})
    target_include_directories(${TARGET} PRIVATE ${Backtrace_INCLUDE_DIR})

    #set(LD_FLAGS_IGNORE_UNDEFINED "-Wl,-undefined,dynamic_lookup")
    #AddFlags(${TARGET} COMPILE_FLAGS ${LD_FLAGS_IGNORE_UNDEFINED})
    #AddFlags(${TARGET} LINK_FLAGS ${LD_FLAGS_IGNORE_UNDEFINED})
    #set(LD_FLAGS_DYNAMIC "-rdynamic")
    #AddFlags(${TARGET} LINK_FLAGS ${LD_FLAGS_DYNAMIC})
    ##AddFlags(${TARGET} COMPILE_FLAGS "-DBOOST_STACKTRACE_USE_ADDR2LINE -g")
    ##AddFlags(${TARGET} LINK_FLAGS "-g")
    #AddFlags(${TARGET} COMPILE_FLAGS "-DBOOST_STACKTRACE_USE_BACKTRACE -g")
    #AddFlags(${TARGET} LINK_FLAGS "-g")

    if (ENABLE_BACKTRACE)
        set(BACKTRACE_SYMS_FLAGS "-g")
        AddFlags(${TARGET} COMPILE_FLAGS ${BACKTRACE_SYMS_FLAGS})
        AddFlags(${TARGET} LINK_FLAGS ${BACKTRACE_SYMS_FLAGS})
    endif()

    #
    # Ensure only symbols we want exported by our LD_PRELOAD library get exported to the application.
    #

    # For our source files, only expose symbols that are marked with RLSCOPE_EXPORT.
    include(GenerateExportHeader)
    generate_export_header(${TARGET})

    RLS_INSTALL(
            FILES ${PROJECT_BINARY_DIR}/${TARGET}_export.h
            LOCAL_INSTALL_DIR include)


    if (WITH_CUDA_LD_PRELOAD)
    #    set(BACKTRACE_SYMS_FLAGS "-g")

        # AddFlags fails here for some reason.
    #    AddFlags(${TARGET} COMPILE_DEFINITIONS "-DWITH_CUDA_LD_PRELOAD")
    #    set(WITH_CUDA_LD_PRELOAD_FLAGS "-DWITH_CUDA_LD_PRELOAD")
    #    AddFlags(${TARGET} COMPILE_DEFINITIONS ${WITH_CUDA_LD_PRELOAD_FLAGS})

    #    get_target_property(OLD_DEFS ${TARGET} COMPILE_DEFINITIONS)
    #    set_target_properties(${TARGET} PROPERTIES COMPILE_DEFINITIONS "${OLD_DEFS} -DWITH_CUDA_LD_PRELOAD")

        target_compile_definitions(${TARGET} PRIVATE "-DWITH_CUDA_LD_PRELOAD")
        target_compile_definitions(${TARGET} PRIVATE "-DMADEUP")

        # Create CUDA_LD_PRELOAD_EXPORT macro for exporting symbols in cuda_ld_preload.cc
        generate_export_header(${TARGET}
                BASE_NAME cuda_ld_preload)
    endif()
    set_target_properties(${TARGET} PROPERTIES CXX_VISIBILITY_PRESET hidden)

    # For static libraries we link to, DON'T expose ANY of their symbols.
    set(HIDE_LINKED_LIB_SYMS_FLAGS "-Wl,--exclude-libs,ALL")
    AddFlags(${TARGET} LINK_FLAGS ${HIDE_LINKED_LIB_SYMS_FLAGS})

    #
    # !!IMPORTANT!!
    #
    # See we re-used some tensorflow code, we also reused libraries like "absl".
    # Sadly, the absl static libraries export weak symbols.
    # When we use LD_PRELOAD, those weak symbols will OVERRIDE the absl library
    # that tensorflow uses internally.
    # This CAN and DOES cause segfaults
    # (when absl versions of lib${TARGET}.so don't match tensorflow.so).
    # What makes it extra fun is that the stacktrace we get tells us NOTHING
    # about this.
    # Also, using "hidden" visibility is INSUFFICIENT for solving this, since
    # static library symbols still get "absorbed" by the final linked
    # shared library.
    # Only way around this is the linker version script.
    #
    # TLDR: we NEED a "linker version script" that explicitly says to make all
    # symbols local, except:
    # ${TARGET}.py API symbols:
    #   - setup, print, set_metadata, enable_tracing, is_enabled, disable_tracing, async_dump, await_dump, record_event, record_overhead_event, record_overhead_event_for_operation, push_operation, pop_operation
    # CUDA API calls we intercept:
    #   - cudaLaunchKernel, cudaMemcpyAsync, cudaMalloc, cudaFree
    #
    # Useful resources:
    # - Stackoverflow question describing our exact issue in simplified form:
    #   https://stackoverflow.com/questions/22102470/link-a-static-library-to-a-shared-library-and-hide-exported-symbols
    # - Linker script syntax:
    #   https://ftp.gnu.org/old-gnu/Manuals/ld-2.9.1/html_node/ld_25.html
    # - Discussion about hidden visibility:
    #   https://labjack.com/news/simple-cpp-symbol-visibility-demo
    #
    set(LD_VERSION_SCRIPT ${PROJECT_SOURCE_DIR}/src/ld.version)
    set(VERSION_LD_SCRIPT_FLAGS "-Wl,--version-script=${LD_VERSION_SCRIPT}")
    AddFlags(${TARGET} LINK_FLAGS "${VERSION_LD_SCRIPT_FLAGS}")
    AddFlags(${TARGET} LINK_DEPENDS "${LD_VERSION_SCRIPT}")

    target_link_libraries(${TARGET} PRIVATE nsync_includes)

    set(COMPILE_FLAGS_HIDDEN_VISIBILITY "-fvisibility=hidden")
    AddFlags(${TARGET} COMPILE_FLAGS ${COMPILE_FLAGS_HIDDEN_VISIBILITY})
endfunction()

function(AddTestAnalysisTarget TARGET)
    add_executable(${TARGET} ${TEST_ANALYSIS_SOURCES})
    AddTensorflowDependencies(${TARGET} FALSE)
    AddUnitTest(${TARGET})
    AddCommonRLSDeps(${TARGET})
endfunction()

function(AddCommonRLSDeps TARGET)
    # This will find the ALREADY INSTALLED libcommon_util.a from previous cmake runs,
    # NOT the one we are building in this run...
#    find_library(common_util_LIBRARY libcommon_util.a)
#    if (NOT common_util_LIBRARY)
#        message(FATAL_ERROR "NOTFOUND: Couldn't find libcommon_util.a")
#    else()
#        message("FOUND: found libcommon_util.a @ ${common_util_LIBRARY}")
#    endif()
#    target_link_libraries(${TARGET} PRIVATE ${common_util_LIBRARY})
    target_link_libraries(${TARGET} PRIVATE common_util)
    target_link_libraries(${TARGET} PRIVATE range_sampling)
    target_link_libraries(${TARGET} PRIVATE rlscope_common)
endfunction()

function(AddGFlagsDeps TARGET)
    find_package(
            gflags REQUIRED
            # GFlags REQUIRED
            # Static linkage to make installing rlscope libs/bins more portable.
            COMPONENTS static)
    message("> GFLAGS_LIBRARIES = ${GFLAGS_LIBRARIES}")
    message("> GFLAGS_INCLUDE_DIRS = ${GFLAGS_INCLUDE_DIRS}")
    target_link_libraries(${TARGET} PUBLIC ${GFLAGS_LIBRARIES})
endfunction()

function(AddRLSAnalyzeTarget TARGET)
    add_executable(${TARGET} ${RLS_ANALYZE_SRCS})
    AddTensorflowDependencies(${TARGET} FALSE)

    AddGFlagsDeps(${TARGET})
    AddCommonRLSDeps(${TARGET})
endfunction()

set(RLS_INSTALL_CONFIGURATIONS Debug Release)
function(RLS_INSTALL)
    set(options)
    # RLSCOPE_INSTALL_DIR
    set(oneValueArgs COMPONENT LOCAL_INSTALL_DIR)
    set(multiValueArgs)
    cmake_parse_arguments(RLS_INSTALL "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    string(FIND "${RLS_INSTALL_KEYWORDS_MISSING_VALUES}" COMPONENT HAS_COMPONENT)
    if (NOT HAS_COMPONENT)
        set(RLS_INSTALL_COMPONENT "rlscope")
    endif()

    if (NOT RLS_INSTALL_LOCAL_INSTALL_DIR)
        message(FATAL_ERROR
                "RLS_INSTALL missing keyword arguments: RLS_INSTALL_LOCAL_INSTALL_DIR")
    endif()

    # if (RLS_INSTALL_KEYWORDS_MISSING_VALUES)
    #     message(FATAL_ERROR
    #             "RLS_INSTALL missing keyword arguments: ${RLS_INSTALL_KEYWORDS_MISSING_VALUES}")
    # endif()

    # INSTALL_FILE:
    #   Path to file we're install(..)-ing
    # LOCAL_INSTALL_DIR:
    #   "include" for header files, "lib" for library files, "bin" for executables
    # RLSCOPE_INSTALL_DIR:
    #   rlscope/include for header files, rlscope/lib for library files, rlscope/bin for executables
    install(
            # [FILES|TARGET] ...
            ${RLS_INSTALL_UNPARSED_ARGUMENTS}
            CONFIGURATIONS ${RLS_INSTALL_CONFIGURATIONS}
            COMPONENT ${COMPONENT}
            DESTINATION ${RLS_INSTALL_LOCAL_INSTALL_DIR}
            )


    # NOTE: I used to install rlscope files into a separate rlscope directory, but it's not needed
    # since I now statically link librlscope.so and avoid third party install(...) rules
    # (see EXCLUDE_FROM_ALL in add_subdirectory calls)
    #
    # install(
    #         # [FILES|TARGET] ...
    #         ${RLS_INSTALL_UNPARSED_ARGUMENTS}
    #         CONFIGURATIONS ${RLS_INSTALL_CONFIGURATIONS}
    #         COMPONENT "rlscope-dir"
    #         DESTINATION ${RLS_INSTALL_RLSCOPE_INSTALL_DIR}
    #         )

endfunction()

#
# Library targets: ./src/libs
# e.g., common_util, range_sampling
#

include(libs.cmake)
#return()

message("> common_util_GTEST_DIRS = ${common_util_GTEST_DIRS}")

set(TEST_ANALYSIS_SOURCES)
if (RLS_TEST_SRCS)
    set(TEST_ANALYSIS_SOURCES
            ${COMMON_SOURCES}
            ${ANALYSIS_SOURCES}
            ${RLS_TEST_SRCS}
            ${BACKWARD_ENABLE}
            ${RLSCOPE_PROTO_SRCS} ${RLSCOPE_PROTO_HDRS})
    list(FILTER TEST_ANALYSIS_SOURCES EXCLUDE REGEX "/c_test_util.cc")
    if (DEBUG_CMAKE_VERBOSE)
        message("> TEST_ANALYSIS_SOURCES = ${TEST_ANALYSIS_SOURCES}")
    endif()
endif()


#
# Create targets.
#

set(BINARIES)
set(LIBRARIES)
if (WITH_CXX_ANALYSIS)
    AddRLSAnalyzeTarget(rls-analyze)
    list(APPEND BINARIES rls-analyze)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        # In debug mode, include 'rls-test' unit tests as part of the binary archive (i.e., CPack package).
        if (TEST_ANALYSIS_SOURCES)
            AddTestAnalysisTarget(rls-test)
            RLS_INSTALL(
                    TARGETS rls-test
                    LOCAL_INSTALL_DIR bin
                    # COMPONENT "rlscope-test"
            )
            # list(APPEND BINARIES rls-test)
        endif()
    endif()
endif()
AddRLScopeTarget(rlscope)
list(APPEND LIBRARIES rlscope)

RLS_INSTALL(
        FILES src/public_headers/rlscope_c_api.h
        LOCAL_INSTALL_DIR include)

#
# Install targets.
#

if (BINARIES)
    message("> INSTALL BINARIES: ${BINARIES}")
    RLS_INSTALL(
            TARGETS ${BINARIES}
            LOCAL_INSTALL_DIR bin)
endif()


if (LIBRARIES)
    RLS_INSTALL(
            TARGETS ${LIBRARIES}
            LOCAL_INSTALL_DIR lib)
endif()

set(CPACK_GENERATOR "TGZ")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "James Gleeson") # required
set(CPACK_PACKAGE_VENDOR "James Gleeson") # required
set(CPACK_COMPONENTS_ALL "rlscope")
#cpack_add_component("rlscope-dir"
#        DISPLAY_NAME "RL-Scope system libraries/binaries/includes"
#        DESCRIPTION "RL-Scope install files in the standard platform-specific /usr/{lib,bin,include} directories"
#        #        [HIDDEN | REQUIRED | DISABLED ]
#        #        [GROUP group]
#        #        [DEPENDS comp1 comp2 ... ]
#        #        [INSTALL_TYPES type1 type2 ... ]
#        #        [DOWNLOADED]
#        #        [ARCHIVE_FILE filename]
#        )
# TODO: derive this from the git tag (e.g., v1.0.0) from which it was built?
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "0")
include(CPack)

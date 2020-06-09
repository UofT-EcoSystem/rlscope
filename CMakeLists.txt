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
project(iml LANGUAGES ${LANGUAGES})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
#message("> CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}")

include(LibFindMacros)

if(DEFINED ENV{IML_INSTALL_PREFIX})
    message("> Building for docker:")
    set(IML_INSTALL_PREFIX $ENV{IML_INSTALL_PREFIX})
else()
    message("> Building for host")
    set(IML_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/local.host")
endif()
message("  IML_INSTALL_PREFIX=${IML_INSTALL_PREFIX}")
# Override system installation of libraries (e.g., gflags)
link_directories(BEFORE ${IML_INSTALL_PREFIX}/lib)
include_directories(BEFORE ${IML_INSTALL_PREFIX}/include)

# Use pkg-config to get hints about paths
#libfind_pkg_check_modules(CUDAProfilingAPI_PKGCONF CUDAProfilingAPI)

find_package(CUDAProfilingAPI REQUIRED)
#include_directories(${CUDAProfilingAPI_INCLUDE_DIR})
# Gets included by libraries that require it (e.g., profilerhost_util)
#link_libraries(${CUDAProfilingAPI_LIBRARY})

# https://stackoverflow.com/questions/32469953/why-is-cmake-designed-so-that-it-removes-runtime-path-when-installing
#list(APPEND CMAKE_INSTALL_RPATH ${BOOST_LIBRARYDIR})
# NOTE: This isn't enough to keep BOOST_LIBRARYDIR in RUNPATH for some reason.
# See CMAKE_INSTALL_RPATH related line below.
# This line does however cause CUDA/CUPTI directories to be kept oddly (which presumably we want...).
#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

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
if(MINGW OR CYGWIN OR WIN32 OR MSVC OR CLION)
    #    message(FATAL_ERROR "WHY")
    message("> Detected Windows development machine")
    set(IS_WINDOWS ON)
else()
    message("> Detected Unix setup (build environment)")
    set(IS_WINDOWS OFF)
endif()
if(IS_WINDOWS OR CLION)
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
        protobuf-${PROTOBUF_VERSION}
        # libpqxx
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
add_subdirectory(third_party/nsync)

# Third party: abseil-cpp
# https://github.com/abseil/abseil-cpp/blob/master/CMake/README.md
set(ABSL_RUN_TESTS OFF)
set(BUILD_TESTING OFF)
add_subdirectory(third_party/abseil-cpp)
unset(BUILD_TESTING)

# Third party: backward-cpp
add_subdirectory(third_party/backward-cpp)

# Third party: eigen
find_package(Eigen3 REQUIRED)

# Third party: json
set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(third_party/json)

#
# Compiler definitions.
#
set(IGNORE_ERRORS "-Wno-unused -Wno-unknown-pragmas")
set(ADD_WARNINGS_FLAGS "-Wuninitialized")
#set(CMAKE_CXX_STANDARD 11)
#-std=c++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ADD_WARNINGS_FLAGS} -Wall -Werror ${IGNORE_ERRORS} ")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -demangle")

include_directories(
        ./
        ./src
        ./src/libs/include
        ./src/libs
        ./external_libs/include
)

set(SOURCES)
# $ find tensorflow -type d
set(TF_SRC_DIRS
        tensorflow
        tensorflow/c
        tensorflow/core
        tensorflow/core/lib
        tensorflow/core/lib/hash
        tensorflow/core/lib/core
        tensorflow/core/lib/io
        tensorflow/core/lib/strings
        tensorflow/core/lib/gtl
        tensorflow/core/platform
        tensorflow/core/platform/posix
        tensorflow/core/platform/default
        )

set(ANALYSIS_SRC_DIRS
        ./src/analysis
        )

set(COMMON_SRC_DIRS
        ./src/common
        )

set(TEST_ANALYSIS_SRC_DIRS
        ./test/analysis
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
        ${TF_SRC_DIRS}
        )

#
# Gather paths of source files for creating RLScope targets (i.e., binaries, libraries)
#

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

gather_sources(ANALYSIS_SOURCES ANALYSIS_SRC_DIRS)
gather_sources(COMMON_SOURCES COMMON_SRC_DIRS)
gather_sources(TEST_ANALYSIS_SOURCES TEST_ANALYSIS_SRC_DIRS)
set(ANALYSIS_SOURCES
        ./src/cuda_api_profiler/generic_logging.cc
        ./src/cuda_api_profiler/generic_logging.h
        ${ANALYSIS_SOURCES})
gather_sources(SOURCES SRC_DIRS)
gather_sources(TF_SOURCES TF_SRC_DIRS)

set(CUDA_PROF_DIRS
        ./src/cuda_api_profiler
        )
include_directories(/usr/local/cuda/extras/CUPTI/include)
include_directories(/usr/local/cuda/include)

gather_sources(CUDA_PROF_SRCS CUDA_PROF_DIRS)

# https://github.com/bombela/backward-cpp/tree/84ae4f5e80381aca765a0810d4c811acae3cd7c7#configuration--dependencies

if (DEBUG_CMAKE_VERBOSE)
    message("> CUDA_PROF_SRCS = ${CUDA_PROF_SRCS}")
    message("> TF_SOURCES = ${TF_SOURCES}")
    message("> ANALYSIS_SOURCES = ${ANALYSIS_SOURCES}")
    message("> COMMON_SOURCES = ${COMMON_SOURCES}")
    #message("> TEST_ANALYSIS_SOURCES = ${TEST_ANALYSIS_SOURCES}")
    message("> PROTO_SRCS = ${PROTO_SRCS}")
    message("> PROTO_HDRS = ${PROTO_HDRS}")
    message("> BACKWARD_ENABLE = ${BACKWARD_ENABLE}")
endif()


# With PROTOBUF_GENERATE_CPP_APPEND_PATH=OFF
#     In CMake 3.5.1 (Ubuntu 16.04) this makes the
#     generated protobuf .pb.cc and .pb.h maintain their original directory structure:
#       iml/build/tensorflow/core/lib/core/error_codes.pb.h
#     Without this, cmake's FindProtobuf module will still all generated files in the iml/build directory:
#       iml/build/error_codes.pb.h
# HOWEVER; this no longer works in CMake 3.14.5 (Ubuntu 18.04)
#       The directory structure for generated files becomes:
#         iml/build/CMakeFiles/tensorflow/core/lib/core/error_codes.pb.h
#       BUT, PROTO_SRCS still matches this directory structure (non-existent files):
#         iml/build/error_codes.pb.h
# TLDR: Oh well, just generate cpp files in iml/build/*.pb.cpp
# set(PROTOBUF_GENERATE_CPP_APPEND_PATH OFF)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS
        tensorflow/core/lib/core/error_codes.proto
        iml_profiler/protobuf/iml_prof.proto
        iml_profiler/protobuf/pyprof.proto
        )
set(SAMPLE_CUDA_API_SOURCES
        ${TF_SOURCES}
        # Add SIGSEGV handler that prints out stack-trace.
        # NOTE: For some reason this doesn't always work and gdb is required to debug...
        # I think that's because the program being executed overrides the SIGSEGV handler
        # (which is FINE, as long as backward-cpp doesn't override the application handler).
        ${BACKWARD_ENABLE}
        # NOTE: we should probably put "common" stuff between 'analysis' and libsample in the common directory,
        # instead of pulling in all of the analysis sources.
        ${COMMON_SOURCES}
        ${PROTO_SRCS} ${PROTO_HDRS}
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
            #            ${TF_SOURCES}
            ${BACKWARD_ENABLE}
            ${PROTO_SRCS} ${PROTO_HDRS} src/analysis/lru_cache.cc src/analysis/sample_periods.cc src/analysis/sample_periods.h)
    list(FILTER RLS_ANALYZE_SRCS EXCLUDE REGEX "/c_test_util.cc")
    if (DEBUG_CMAKE_VERBOSE)
        message("> RLS_ANALYZE_SRCS = ${RLS_ANALYZE_SRCS}")
    endif()

    set(TEST_ANALYSIS_SOURCES
            ${COMMON_SOURCES}
            ${ANALYSIS_SOURCES}
            ${TEST_ANALYSIS_SOURCES}
            #            ${TF_SOURCES}
            ${BACKWARD_ENABLE}
            ${PROTO_SRCS} ${PROTO_HDRS})
    list(FILTER TEST_ANALYSIS_SOURCES EXCLUDE REGEX "/c_test_util.cc")
    if (DEBUG_CMAKE_VERBOSE)
        message("> TEST_ANALYSIS_SOURCES = ${TEST_ANALYSIS_SOURCES}")
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

function(AddUnitTest TARGET)
    find_package(GTest REQUIRED)
    # message("> GTEST_LIBRARIES = ${GTEST_LIBRARIES}")
    # message("> GTEST_INCLUDE_DIRS = ${GTEST_INCLUDE_DIRS}")
    target_include_directories(${TARGET} PRIVATE ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${TARGET} PRIVATE ${GTEST_LIBRARIES})
endfunction()

function(AddCUPTIDependencies TARGET)
    find_package(Libcupti REQUIRED)
    target_link_libraries(${TARGET} PUBLIC ${Libcupti_LIBRARY})
    # https://stackoverflow.com/questions/26243169/cmake-target-include-directories-meaning-of-scope
    # NOTE: I still don't fully understand the PUBLIC/PRIVATE/INTERFACE scope here...
    target_include_directories(${TARGET} PRIVATE ${Libcupti_INCLUDE_DIR})
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

    target_link_libraries(${TARGET} PRIVATE ${Protobuf_LIBRARIES})
    #message("> PROTO_SRCS = ${PROTO_SRCS}")
    #message("> PROTO_HDRS = ${PROTO_HDRS}")
    target_link_libraries(${TARGET} PRIVATE
            absl::base
            absl::strings
            absl::memory)
    target_link_libraries(${TARGET} PRIVATE nsync)

    IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("> Enable spdlog logging for target=${TARGET}")

        get_target_property(CUR_FLAG ${TARGET} COMPILE_DEFINITIONS)
        set_target_properties(${TARGET} PROPERTIES COMPILE_DEFINITIONS "${CUR_FLAG};SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG")
        get_target_property(CUR_FLAG ${TARGET} COMPILE_DEFINITIONS)
    ELSE()
        message("> SKIP: spdlog logging for target=${TARGET}")

        get_target_property(CUR_FLAG ${TARGET} COMPILE_DEFINITIONS)
        set_target_properties(${TARGET} PROPERTIES COMPILE_DEFINITIONS "${CUR_FLAG};SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_OFF")
        get_target_property(CUR_FLAG ${TARGET} COMPILE_DEFINITIONS)
    ENDIF()

    target_link_libraries(${TARGET} PRIVATE nlohmann_json::nlohmann_json)

    # Simple header-only thread pool library.
    set(CTPL_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/third_party/CTPL)
    target_include_directories(${TARGET} PRIVATE ${CTPL_INCLUDE_DIRS})

    target_link_libraries(${TARGET} PRIVATE boost_filesystem)

    target_link_libraries(${TARGET} PRIVATE nsync_includes)

    find_package(spdlog REQUIRED)
    target_link_libraries(${TARGET} PRIVATE spdlog::spdlog)

    target_link_libraries(${TARGET} PRIVATE nsync_cpp)
    target_include_directories(${TARGET} PRIVATE ${double_conversion_INCLUDE_DIR})
    target_link_libraries(${TARGET} PRIVATE ${double_conversion_LIBRARIES})
    target_link_libraries(${TARGET} PRIVATE Eigen3::Eigen)
    target_link_libraries(${TARGET} PRIVATE "-ldl")
    if(INCLUDE_CUDA)
        find_package(Libcupti REQUIRED)
        target_link_libraries(${TARGET} PUBLIC ${Libcupti_LIBRARY})
        # https://stackoverflow.com/questions/26243169/cmake-target-include-directories-meaning-of-scope
        # NOTE: I still don't fully understand the PUBLIC/PRIVATE/INTERFACE scope here...
        target_include_directories(${TARGET} PRIVATE ${Libcupti_INCLUDE_DIR})

        if (NOT WITH_CUDA_LD_PRELOAD)
            # CUDA_LIBRARIES = /usr/local/cuda/lib64/libcudart_static.a;-pthread;dl;/usr/lib/x86_64-linux-gnu/librt.so
            # message("> CUDA_LIBRARIES = ${CUDA_LIBRARIES}")
            target_link_libraries(${TARGET} PUBLIC ${CUDA_LIBRARIES})
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
    add_dependencies(${TARGET} double_conversion)
    # This will add libraries, definitions and include directories needed by backward
    # by setting each property on the target.
    add_backward(${TARGET})

    target_link_libraries(${TARGET} PRIVATE boost_filesystem)

    target_link_libraries(${TARGET} PRIVATE nlohmann_json::nlohmann_json)

    target_link_libraries(${TARGET} PRIVATE ${Protobuf_LIBRARIES})
    #message("> PROTO_SRCS = ${PROTO_SRCS}")
    #message("> PROTO_HDRS = ${PROTO_HDRS}")
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
    find_package(Libcupti REQUIRED)
    target_link_libraries(${TARGET} PUBLIC ${Libcupti_LIBRARY})
    # https://stackoverflow.com/questions/26243169/cmake-target-include-directories-meaning-of-scope
    # NOTE: I still don't fully understand the PUBLIC/PRIVATE/INTERFACE scope here...
    target_include_directories(${TARGET} PRIVATE ${Libcupti_INCLUDE_DIR})

    if (NOT WITH_CUDA_LD_PRELOAD)
        # CUDA_LIBRARIES = /usr/local/cuda/lib64/libcudart_static.a;-pthread;dl;/usr/lib/x86_64-linux-gnu/librt.so
        # message("> CUDA_LIBRARIES = ${CUDA_LIBRARIES}")
        target_link_libraries(${TARGET} PUBLIC ${CUDA_LIBRARIES})
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

    # For our source files, only expose symbols that are marked with SAMPLE_CUDA_API_EXPORT.
    include(GenerateExportHeader)
    generate_export_header(${TARGET})

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
endfunction()

function(AddRLSAnalyzeTarget TARGET)
    add_executable(${TARGET} ${RLS_ANALYZE_SRCS})
    AddTensorflowDependencies(${TARGET} FALSE)

    find_package(GFlags REQUIRED)
    # message("> GFLAGS_LIBRARIES = ${GFLAGS_LIBRARIES}")
    # message("> GFLAGS_INCLUDE_DIRS = ${GFLAGS_INCLUDE_DIRS}")
    target_link_libraries(${TARGET} PUBLIC ${GFLAGS_LIBRARIES})
endfunction()

#
# Create targets.
#

if (WITH_CXX_ANALYSIS)
    AddRLSAnalyzeTarget(rls-analyze)
    AddTestAnalysisTarget(rls-test)
endif()
AddRLScopeTarget(rlscope)


#
# Library targets: ./src/libs
# e.g., common_util, range_sampling
#

include(libs.cmake)

#
# Install targets.
#

set(BINARIES
        rls-analyze
        rls-test
        )
install(TARGETS ${BINARIES}
        CONFIGURATIONS Debug
        RUNTIME DESTINATION bin)
install(TARGETS ${BINARIES}
        CONFIGURATIONS Release
        RUNTIME DESTINATION bin)

set(LIBRARIES
        rlscope
        )
install(TARGETS ${LIBRARIES}
        CONFIGURATIONS Debug
        RUNTIME DESTINATION lib)
install(TARGETS ${LIBRARIES}
        CONFIGURATIONS Release
        RUNTIME DESTINATION lib)

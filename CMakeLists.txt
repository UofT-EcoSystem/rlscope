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

# Need cmake >= 3.8 so that we can do:
#   set(LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(iml)
#https://github.com/robertmaynard/code-samples/blob/master/posts/cmake/CMakeLists.txt

set(ENABLE_BACKTRACE ON CACHE BOOL "Enable pretty-printed stacktrace during LOG(FATAL) or DCHECK failures; requires adding -g debug symbols")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#FIND_PACKAGE(CUDA)

set(ENABLE_DQN_CPP OFF)

set(BOOST_VERSION 1.70.0)
set(BOOST_VERSION_UNDERSCORES 1_70_0)

#set(PROTOBUF_VERSION 3.6.1)
set(PROTOBUF_VERSION 3.9.1)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
#message("> CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}")

if(MINGW OR CYGWIN OR WIN32 OR MSVC OR CLION)
#    message(FATAL_ERROR "WHY")
    message("> Detected Windows development machine")
    set(IS_WINDOWS ON)
else()
    message("> Detected Unix setup (build environment)")
    set(IS_WINDOWS OFF)
endif()

#if (NOT IS_WINDOWS)
#    message("> Add libcupti include")
#    include_directories(SYSTEM /usr/local/cuda/extras/CUPTI/include)
#endif()

if(IS_WINDOWS)
    set(LANGUAGES CXX)
else()
    set(LANGUAGES CXX CUDA)
endif()
project(iml LANGUAGES ${LANGUAGES})

#NOT EXISTS ${PROJECT_SOURCE_DIR}/external_libs/lib
if (NOT IS_WINDOWS AND NOT (
        EXISTS ${PROJECT_SOURCE_DIR}/third_party/json
        AND EXISTS ${PROJECT_SOURCE_DIR}/third_party/abseil-cpp
        AND EXISTS ${PROJECT_SOURCE_DIR}/third_party/nsync
        AND EXISTS ${PROJECT_SOURCE_DIR}/third_party/backward-cpp
        AND EXISTS ${PROJECT_SOURCE_DIR}/third_party/eigen
        AND EXISTS ${PROJECT_SOURCE_DIR}/third_party/protobuf-${PROTOBUF_VERSION}
#        AND EXISTS ${PROJECT_SOURCE_DIR}/third_party/boost_${BOOST_VERSION_UNDERSCORES}/build
        ) )
    message(FATAL_ERROR
            "You need to run setup.sh first:\n"
            "  $ cd ${PROJECT_SOURCE_DIR}\n"
            "  $ bash ./setup.sh")
endif()

if (NOT IS_WINDOWS)
    find_package(GTest REQUIRED)
    include_directories(${GTEST_INCLUDE_DIRS})
endif()

# https://github.com/abseil/abseil-cpp/blob/master/CMake/README.md
set(ABSL_RUN_TESTS OFF)
set(BUILD_TESTING OFF)
add_subdirectory(third_party/abseil-cpp)
unset(BUILD_TESTING)

add_subdirectory(third_party/backward-cpp)

#add_subdirectory(third_party/eigen)
#target_include_directories(sample_cuda_api third_party/eigen)
find_package(Eigen3 REQUIRED
        HINTS third_party/eigen/build/cmake_install)

set(NSYNC_ENABLE_TESTS OFF)
#set(BUILD_SHARED_LIBS ON)
add_subdirectory(third_party/nsync)

if(NOT IS_WINDOWS)
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    add_subdirectory(third_party/json)
endif()

set(IGNORE_ERRORS "-Wno-unused -Wno-unknown-pragmas")
set(ADD_WARNINGS_FLAGS "-Wuninitialized")
set(CMAKE_CXX_STANDARD 11)
#-std=c++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ADD_WARNINGS_FLAGS} -Wall -Werror ${IGNORE_ERRORS} ")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -demangle")

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
        cartpole
        epoch_sec)
set(ALL_TARGETS
        ${EXEC_NAMES}
        ${GFLAGS_EXEC_NAMES}
        py_interface
        )

set(SOURCES)
# ( cd ~/clone/dnn_tensorflow_cpp; find -type f | grep --perl-regexp '\.(c|cc|cpp|cxx)$' | xargs -n1 dirname | sort --unique | grep -v CMake; ) | cboard

#        ./tensorflow/c
#        ./tensorflow/core/platform
#        ./tensorflow/core/platform/default
#        ./tensorflow/core/platform/posix

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
#        tensorflow/core/platform/windows

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

#if(IS_WINDOWS)
#    set(CUPTI_DIRS
#            ./CUPTI/include
#            ./CUPTI/include/GL
#            ./CUPTI/include/openacc
#            ./CUPTI/sample
#            ./CUPTI/sample/activity_trace_async
#            ./CUPTI/sample/callback_event
#            ./CUPTI/sample/callback_metric
#            ./CUPTI/sample/callback_timestamp
#            ./CUPTI/sample/cupti_query
#            ./CUPTI/sample/event_multi_gpu
#            ./CUPTI/sample/event_sampling
#            ./CUPTI/sample/nvlink_bandwidth
#            ./CUPTI/sample/openacc_trace
#            ./CUPTI/sample/pc_sampling
#            ./CUPTI/sample/sass_source_map
#            ./CUPTI/sample/unified_memory
#            )
#    gather_sources(CUPTI_SOURCES CUPTI_DIRS)
#    message("CUPTI_DIRS = ${CUPTI_DIRS}")
#    message("CUPTI_SOURCES = ${CUPTI_SOURCES}")
#    add_executable(cupti_dummy ${CUPTI_SOURCES})
#endif()

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

# https://cmake.org/cmake/help/v3.9/module/FindProtobuf.html
set(Protobuf_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/protobuf-${PROTOBUF_VERSION}/build)

#set(Protobuf_LIBRARY ${Protobuf_ROOT_DIR}/lib/libprotobuf.so)
set(Protobuf_LIBRARY ${Protobuf_ROOT_DIR}/lib/libprotobuf.a)
# HACK: I like to use symlinks, but docker container uses abspath's for mounts
get_filename_component(Protobuf_LIBRARY ${Protobuf_LIBRARY} REALPATH)
message("> ABSOLUTE Protobuf_LIBRARY = ${Protobuf_LIBRARY}")
#set(Protobuf_LIBRARY ${Protobuf_ROOT_DIR}/lib/libprotobuf.a)
set(Protobuf_INCLUDE_DIR ${Protobuf_ROOT_DIR}/include)
set(Protobuf_PROTOC_EXECUTABLE ${Protobuf_ROOT_DIR}/bin/protoc)
find_package(Protobuf REQUIRED)
message("> Protobuf_INCLUDE_DIRS = ${Protobuf_INCLUDE_DIRS}")
message("> Protobuf_LIBRARY = ${Protobuf_LIBRARY}")
message("> Protobuf_LIBRARIES = ${Protobuf_LIBRARIES}")
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# https://cmake.org/cmake/help/v3.9/module/FindProtobuf.html
include(double_conversion)
#add_subdirectory(double_conversion)

#find_package(DoubleConversion REQUIRED)
#message("> DOUBLE_CONVERSION_INCLUDE_DIRS = ${DOUBLE_CONVERSION_INCLUDE_DIRS}")
#message("> DOUBLE_CONVERSION_LIBRARY = ${DOUBLE_CONVERSION_LIBRARY}")

find_package(DoubleConversion REQUIRED)
message("> double_conversion_INCLUDE_DIRS = ${double_conversion_INCLUDE_DIRS}")
#message("> double_conversion_LIBRARY = ${double_conversion_LIBRARY}")
message("> double_conversion_LIBRARIES = ${double_conversion_LIBRARIES}")

#include_directories(${CMAKE_CURRENT_BINARY_DIR})

# NOTE: We need PROTOBUF_GENERATE_CPP_APPEND_PATH=OFF so that the
# generated protobuf .pb.cc and .pb.h maintain their original directory structure:
#   iml/build/tensorflow/core/lib/core/error_codes.pb.h
# Without this, cmake's FindProtobuf module will still all generated files in the iml/build directory:
#   iml/build/error_codes.pb.h
set(PROTOBUF_GENERATE_CPP_APPEND_PATH OFF)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS
        tensorflow/core/lib/core/error_codes.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS
        iml_profiler/protobuf/iml_prof.proto)

#protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS EXPORT_MACRO DLL_EXPORT foo.proto)
#protobuf_generate_python(PROTO_PY foo.proto)

#add_executable(bar bar.cc ${PROTO_SRCS} ${PROTO_HDRS})
#target_link_libraries(bar ${Protobuf_LIBRARIES})

# BLAZ
if (ENABLE_DQN_CPP)
    add_executable(model
            ./src/drivers/run_model.cpp
            ${SOURCES})
    add_executable(cartpole
            ./src/drivers/run_cartpole.cpp
            ${SOURCES})
    add_executable(tests
            ./src/drivers/run_tests.cpp
            ${TEST_SOURCES})
    add_executable(epoch_sec
            ./src/test/epoch_sec.cpp
            #        ${TF_SOURCES}
            )
endif()

set(SAMPLE_CUDA_API_DIRS
        ./src/cuda_api_profiler
        ${TF_SRC_DIRS}
        )
if(IS_WINDOWS)
    include_directories(./windows_includes/CUPTI/include)
    include_directories(./windows_includes/cuda/include)
else()
    include_directories(/usr/local/cuda/extras/CUPTI/include)
    include_directories(/usr/local/cuda/include)
endif()

message("> TF_SOURCES = ${TF_SOURCES}")
gather_sources(SAMPLE_CUDA_API_SOURCES SAMPLE_CUDA_API_DIRS)

if(IS_WINDOWS)
    # Include headers generated from protobuf files.
    include_directories(./build)
endif()

# https://github.com/bombela/backward-cpp/tree/84ae4f5e80381aca765a0810d4c811acae3cd7c7#configuration--dependencies
add_library(sample_cuda_api SHARED
        ${SAMPLE_CUDA_API_SOURCES}
        ${TF_SOURCES}
        ${PROTO_SRCS} ${PROTO_HDRS}
        # Add SIGSEGV handler that prints out stack-trace.
        # NOTE: For some reason this doesn't always work and gdb is required to debug...
        # I think that's because the program being executed overrides the SIGSEGV handler
        # (which is FINE, as long as backward-cpp doesn't override the application handler).
        ${BACKWARD_ENABLE}
        )
add_dependencies(sample_cuda_api double_conversion)
# This will add libraries, definitions and include directories needed by backward
# by setting each property on the target.
add_backward(sample_cuda_api)

target_link_libraries(sample_cuda_api PRIVATE ${Protobuf_LIBRARIES})
message("> PROTO_SRCS = ${PROTO_SRCS}")
message("> PROTO_HDRS = ${PROTO_HDRS}")
#absl::synchronization
target_link_libraries(sample_cuda_api PRIVATE
        absl::base
        absl::strings
        absl::memory)
target_link_libraries(sample_cuda_api PRIVATE nsync)
target_link_libraries(sample_cuda_api PRIVATE nsync_cpp)
target_link_libraries(sample_cuda_api PUBLIC pthread)
#target_include_directories(sample_cuda_api PRIVATE ${DOUBLE_CONVERSION_INCLUDE_DIRS})
target_include_directories(sample_cuda_api PRIVATE ${double_conversion_INCLUDE_DIR})
#target_link_libraries(sample_cuda_api ${DOUBLE_CONVERSION_LIBRARY})
#target_link_libraries(sample_cuda_api ${double_conversion_LIBRARIES})
#target_link_libraries(sample_cuda_api double_conversion)
target_link_libraries(sample_cuda_api PRIVATE ${double_conversion_LIBRARIES})
target_link_libraries(sample_cuda_api PRIVATE Eigen3::Eigen)
if(NOT IS_WINDOWS)
    find_package(Libcupti REQUIRED)
    target_link_libraries(sample_cuda_api PUBLIC ${Libcupti_LIBRARY})
    # https://stackoverflow.com/questions/26243169/cmake-target-include-directories-meaning-of-scope
    # NOTE: I still don't fully understand the PUBLIC/PRIVATE/INTERFACE scope here...
    target_include_directories(sample_cuda_api PRIVATE ${Libcupti_INCLUDE_DIR})
#    add_subdirectory(third_party/boost_${BOOST_VERSION_UNDERSCORES})

#    set(BOOST_ROOT ${CMAKE_SOURCE_DIR}/third_party/boost_${BOOST_VERSION_UNDERSCORES}/build)
#    set(BOOST_INCLUDEDIR ${CMAKE_SOURCE_DIR}/third_party/boost_${BOOST_VERSION_UNDERSCORES}/build/include)
#    set(BOOST_LIBRARYDIR ${CMAKE_SOURCE_DIR}/third_party/boost_${BOOST_VERSION_UNDERSCORES}/build/lib)
#    set(Boost_NO_SYSTEM_PATHS ON)
#    find_package(Boost ${BOOST_VERSION} REQUIRED
##            COMPONENTS stacktrace
#            COMPONENTS stacktrace_backtrace
##            COMPONENTS stacktrace_addr2line
##            HINTS ${CMAKE_SOURCE_DIR}/third_party/boost_${BOOST_VERSION_UNDERSCORES}/build
#            )
#
#    message("Boost_INCLUDE_DIR = ${Boost_INCLUDE_DIR}")
#    message("Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}")
#
#    message("Boost_LIBRARY_DIRS = ${Boost_LIBRARY_DIRS}")
#    message("Boost_LIBRARY = ${Boost_LIBRARY}")
#    message("Boost_LIBRARIES = ${Boost_LIBRARIES}")
#
#    target_link_libraries(sample_cuda_api ${Boost_LIBRARIES})
#    target_link_directories(sample_cuda_api PRIVATE ${Boost_LIBRARY_DIRS})
#
#    target_include_directories(sample_cuda_api PRIVATE ${Boost_INCLUDE_DIRS})
##    target_link_libraries(sample_cuda_api ${Boost_LIBRARIES})

    find_package(Backtrace REQUIRED)
    target_link_libraries(sample_cuda_api PUBLIC ${Backtrace_LIBRARY})
    target_include_directories(sample_cuda_api PRIVATE ${Backtrace_INCLUDE_DIR})

endif()

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
    set_target_properties(${TARGET} PROPERTIES ${PROPERTY} ${OLD_FLAGS} )
endfunction()

#set(LD_FLAGS_IGNORE_UNDEFINED "-Wl,-undefined,dynamic_lookup")
#AddFlags(sample_cuda_api COMPILE_FLAGS ${LD_FLAGS_IGNORE_UNDEFINED})
#AddFlags(sample_cuda_api LINK_FLAGS ${LD_FLAGS_IGNORE_UNDEFINED})
#set(LD_FLAGS_DYNAMIC "-rdynamic")
#AddFlags(sample_cuda_api LINK_FLAGS ${LD_FLAGS_DYNAMIC})
##AddFlags(sample_cuda_api COMPILE_FLAGS "-DBOOST_STACKTRACE_USE_ADDR2LINE -g")
##AddFlags(sample_cuda_api LINK_FLAGS "-g")
#AddFlags(sample_cuda_api COMPILE_FLAGS "-DBOOST_STACKTRACE_USE_BACKTRACE -g")
#AddFlags(sample_cuda_api LINK_FLAGS "-g")

if (ENABLE_BACKTRACE)
    set(BACKTRACE_SYMS_FLAGS "-g")
    AddFlags(sample_cuda_api COMPILE_FLAGS ${BACKTRACE_SYMS_FLAGS})
    AddFlags(sample_cuda_api LINK_FLAGS ${BACKTRACE_SYMS_FLAGS})
endif()

#
# Ensure only symbols we want exported by our LD_PRELOAD library get exported to the application.
#

# For our source files, only expose symbols that are marked with SAMPLE_CUDA_API_EXPORT.
include(GenerateExportHeader)
generate_export_header(sample_cuda_api)
set_target_properties(sample_cuda_api PROPERTIES CXX_VISIBILITY_PRESET hidden)

# For static libraries we link to, DON'T expose ANY of their symbols.
set(HIDE_LINKED_LIB_SYMS_FLAGS "-Wl,--exclude-libs,ALL")
#AddFlags(sample_cuda_api COMPILE_FLAGS ${HIDE_LINKED_LIB_SYMS_FLAGS})
AddFlags(sample_cuda_api LINK_FLAGS ${HIDE_LINKED_LIB_SYMS_FLAGS})


##https://stackoverflow.com/questions/36662920/xcode-clang-link-build-dynamic-framework-or-dylib-not-embed-dependencies
#set(TARGET "sample_cuda_api")
#set(LD_IGNORE_UNDEFINED_SYM_FLAGS "-Wl,-undefined,dynamic_lookup")
#get_target_property(NEW_FLAGS ${TARGET} COMPILE_FLAGS)
#if(NEW_FLAGS STREQUAL "NEW_FLAGS-NOTFOUND")
#  SET(NEW_FLAGS "") # Set to empty string
#else()
#  SET(NEW_FLAGS "${NEW_FLAGS} ") # A space to cleanly separate from existing content
#endif()
## Append our values
#SET(NEW_FLAGS "${NEW_FLAGS}${LD_IGNORE_UNDEFINED_SYM_FLAGS}" )
#set_target_properties(${TARGET} PROPERTIES COMPILE_FLAGS ${NEW_FLAGS} )



#message("> NSYNC_INCLUDES = ${NSYNC_INCLUDES}")
#target_include_directories(sample_cuda_api ${NSYNC_INCLUDES})

add_library(nsync_includes INTERFACE)
target_include_directories(nsync_includes INTERFACE third_party/nsync/public)
target_link_libraries(sample_cuda_api PRIVATE nsync_includes)


#./test/py_interface/py_interface_src.cpp
if (ENABLE_DQN_CPP)
    add_library(py_interface SHARED
            ${PY_INTERFACE_LIB_SOURCES}
    #        ${TF_SOURCES}
            )
endif()

if (ENABLE_DQN_CPP)
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
endif()

if (ENABLE_DQN_CPP)
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
endif()

if (ENABLE_DQN_CPP)
    if(NOT IS_WINDOWS)
        find_library(GFLAGS_LIB gflags REQUIRED)
        message("> GFLAGS_LIB = ${GFLAGS_LIB}")
        foreach(executable ${GFLAGS_EXEC_NAMES})
            target_link_libraries(${executable} ${GFLAGS_LIB})
        endforeach()
    endif()
endif()

#if(NOT IS_WINDOWS)
#    find_library(GTEST_LIB gtest REQUIRED)
#    message("> GTEST_LIB = ${GTEST_LIB}")
#    target_link_libraries(tests ${GTEST_LIB})
#endif()

if (ENABLE_DQN_CPP)
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
endif()

if (ENABLE_DQN_CPP)
    foreach(trg ${ALL_TARGETS})
        target_compile_features(${trg} PUBLIC cxx_std_11)
    endforeach()
endif()

configure_file(normalized_car_features.csv ${CMAKE_CURRENT_BINARY_DIR}/normalized_car_features.csv COPYONLY)
if(MSVC)
    target_compile_definitions(main PRIVATE COMPILER_MSVC)
endif(MSVC)
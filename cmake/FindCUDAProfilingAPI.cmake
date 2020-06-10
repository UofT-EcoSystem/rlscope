# - Try to find CUDAProfilingAPI
# Once done, this will define
#
#  CUDAProfilingAPI_FOUND - system has CUDAProfilingAPI
#  CUDAProfilingAPI_INCLUDE_DIRS - the CUDAProfilingAPI include directories
#  CUDAProfilingAPI_LIBRARIES - link these to use CUDAProfilingAPI

include(LibFindMacros)

# Dependencies
#libfind_package(CUDAProfilingAPI CUDA)
find_package(CUDA)

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(CUDAProfilingAPI_PKGCONF CUDAProfilingAPI)

set(CUPTI_LIB64_DIR ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64)
# ls -lh /usr/local/cuda-10.1/extras/CUPTI/lib64/libnvperf_*
# -rw-r--r-- 1 root root 9.3M Aug  9  2019 /usr/local/cuda-10.1/extras/CUPTI/lib64/libnvperf_host.so
# -rw-r--r-- 1 root root  15M Aug  9  2019 /usr/local/cuda-10.1/extras/CUPTI/lib64/libnvperf_host_static.a
# -rw-r--r-- 1 root root 2.3M Aug  9  2019 /usr/local/cuda-10.1/extras/CUPTI/lib64/libnvperf_target.so

set(CUDA_TARGETS_LIB64_DIR ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib)
# $ ls targets/x86_64-linux/lib/libnvperf_* -l
# -rw-r--r-- 1 root root  9745048 Nov 13  2019 targets/x86_64-linux/lib/libnvperf_host.so
# -rw-r--r-- 1 root root 14776200 Nov 13  2019 targets/x86_64-linux/lib/libnvperf_host_static.a
# -rw-r--r-- 1 root root  2353880 Nov 13  2019 targets/x86_64-linux/lib/libnvperf_target.so

set(CUPTI_INCLUDE_DIR ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include)
# $ ls -lh /usr/local/cuda-10.1/extras/CUPTI/include/nvperf_*
# -rw-r--r-- 1 root root 12K Aug  9  2019 /usr/local/cuda-10.1/extras/CUPTI/include/nvperf_cuda_host.h
# -rw-r--r-- 1 root root 70K Aug  9  2019 /usr/local/cuda-10.1/extras/CUPTI/include/nvperf_host.h
# -rw-r--r-- 1 root root 19K Aug  9  2019 /usr/local/cuda-10.1/extras/CUPTI/include/nvperf_target.h

set(CUDA_TARGETS_INCLUDE_DIR ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include)
# $ ls targets/x86_64-linux/include/*nvperf* -l
# -rw-r--r-- 1 root root 11948 Nov 13  2019 targets/x86_64-linux/include/nvperf_cuda_host.h
# -rw-r--r-- 1 root root 70679 Nov 13  2019 targets/x86_64-linux/include/nvperf_host.h
# -rw-r--r-- 1 root root 20300 Nov 13  2019 targets/x86_64-linux/include/nvperf_target.h

macro(TRY_FIND_PATH PATH_VAR HINT_FILE TRY_DIRECTORIES)
    # set(TRY_DIRECTORIES ${CUPTI_INCLUDE_DIR} ${CUDA_TARGETS_INCLUDE_DIR})
    foreach(include_path IN LISTS ${TRY_DIRECTORIES})
        find_path(${PATH_VAR}
                NAMES ${HINT_FILE}
                PATHS ${CUDAProfilingAPI_PKGCONF_INCLUDE_DIRS}
                HINTS ${include_path})
    endforeach()
    if (NOT ${PATH_VAR} AND CUDAProfilingAPI_FIND_REQUIRED)
        message("Couldn't find ${HINT_FILE}; tried:")
        foreach(include_path IN LISTS ${TRY_DIRECTORIES})
            message("  ${include_path}")
        endforeach()
        message(FATAL_ERROR)
    endif()
endmacro()

macro(TRY_FIND_LIB LIB_ALIAS LIB_VAR HINT_LIB TRY_DIRECTORIES)
    if (${LIB_VAR})
        # Already found.
        return()
    endif()
    foreach(lib_path IN LISTS ${TRY_DIRECTORIES})
        find_library(${LIB_VAR}
                NAMES ${HINT_LIB}
                HINTS ${lib_path})
    endforeach()
    if (NOT ${LIB_VAR} AND CUDAProfilingAPI_FIND_REQUIRED)
        message("Couldn't find ${HINT_LIB}; tried:")
        foreach(lib_path IN LISTS ${TRY_DIRECTORIES})
            message("  ${lib_path}")
        endforeach()
        message(FATAL_ERROR)
    endif()
endmacro()

set(nvperf_host_INCLUDE_DIR_TRY ${CUPTI_INCLUDE_DIR} ${CUDA_TARGETS_INCLUDE_DIR})
TRY_FIND_PATH(nvperf_host_INCLUDE_DIR nvperf_host.h nvperf_host_INCLUDE_DIR_TRY)

set(nvperf_target_INCLUDE_DIR_TRY ${CUPTI_INCLUDE_DIR} ${CUDA_TARGETS_INCLUDE_DIR})
TRY_FIND_PATH(nvperf_target_INCLUDE_DIR nvperf_target.h nvperf_target_INCLUDE_DIR_TRY)

set(nvtx_INCLUDE_DIR_TRY ${CUPTI_INCLUDE_DIR} ${CUDA_TARGETS_INCLUDE_DIR})
TRY_FIND_PATH(nvtx_INCLUDE_DIR nvToolsExt.h nvtx_INCLUDE_DIR_TRY)

set(CUPTI_INCLUDE_DIR_TRY ${CUPTI_INCLUDE_DIR} ${CUDA_TARGETS_INCLUDE_DIR})
TRY_FIND_PATH(cupti_INCLUDE_DIR cupti.h CUPTI_INCLUDE_DIR_TRY)


set(nvperf_host_LIBRARY_TRY ${CUPTI_LIB64_DIR} ${CUDA_TARGETS_LIB64_DIR})
TRY_FIND_LIB(nvperf_host nvperf_host_LIBRARY nvperf_host nvperf_host_LIBRARY_TRY)

set(nvperf_target_LIBRARY_TRY ${CUPTI_LIB64_DIR} ${CUDA_TARGETS_LIB64_DIR})
TRY_FIND_LIB(nvperf_target nvperf_target_LIBRARY nvperf_target nvperf_target_LIBRARY_TRY)

set(nvtx_LIBRARY_TRY ${CUPTI_LIB64_DIR} ${CUDA_TARGETS_LIB64_DIR})
TRY_FIND_LIB(nvtx nvtx_LIBRARY nvToolsExt nvtx_LIBRARY_TRY)

set(cupti_LIBRARY_TRY ${CUPTI_LIB64_DIR} ${CUDA_TARGETS_LIB64_DIR})
TRY_FIND_LIB(cupti cupti_LIBRARY cupti cupti_LIBRARY_TRY)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(CUDAProfilingAPI_PROCESS_INCLUDES
        ${nvperf_host_INCLUDE_DIR}
        ${nvperf_target_INCLUDE_DIR}
        ${nvtx_INCLUDE_DIR}
        ${cupti_INCLUDE_DIR}
        )
message("> CUDAProfilingAPI_PROCESS_INCLUDES = ${CUDAProfilingAPI_PROCESS_INCLUDES}")
set(CUDAProfilingAPI_PROCESS_LIBS
        ${nvperf_host_LIBRARY}
        ${nvperf_target_LIBRARY}
        ${nvtx_LIBRARY}
        ${cupti_LIBRARY}
        )
# This doesn't work for some reason...
#message("> CUDAProfilingAPI_PROCESS_LIBS = ${CUDAProfilingAPI_PROCESS_LIBS}")
#message("> CUDAProfilingAPI_FIND_REQUIRED = ${CUDAProfilingAPI_FIND_REQUIRED}")
#libfind_process(CUDAProfilingAPI)

set(CUDAProfilingAPI_LIBRARIES ${CUDAProfilingAPI_PROCESS_LIBS} CACHE STRING "CUDA profiling libs")
set(CUDAProfilingAPI_INCLUDE_DIRS ${CUDAProfilingAPI_PROCESS_INCLUDES} CACHE STRING "CUDA profiling libs")
set(CUDAProfilingAPI_FOUND TRUE CACHE BOOL "CUDA profiling libs")

message("> CUDAProfilingAPI_LIBRARIES = ${CUDAProfilingAPI_LIBRARIES}")
message("> CUDAProfilingAPI_INCLUDE_DIRS = ${CUDAProfilingAPI_INCLUDE_DIRS}")
find_package_handle_standard_args(CUDAProfilingAPI
        REQUIRED_VARS CUDAProfilingAPI_LIBRARIES CUDAProfilingAPI_INCLUDE_DIRS)


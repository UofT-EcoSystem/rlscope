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

macro(TRY_FIND_LIB LIB_VAR HINT_LIB TRY_DIRECTORIES)
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

#set(CUPTI_INCLUDE_DIR_TRY ${CUPTI_INCLUDE_DIR} ${CUDA_TARGETS_INCLUDE_DIR})
#TRY_FIND_PATH(CUDAProfilingAPI_INCLUDE_DIR nvperf_host.h CUPTI_INCLUDE_DIR_TRY)

#set(TRY_DIRECTORIES ${CUPTI_INCLUDE_DIR} ${CUDA_TARGETS_INCLUDE_DIR})
#foreach(include_path IN LISTS TRY_DIRECTORIES)
#    find_path(CUDAProfilingAPI_INCLUDE_DIR
#            NAMES nvperf_host.h
#            PATHS ${CUDAProfilingAPI_PKGCONF_INCLUDE_DIRS}
#            HINTS ${include_path})
#endforeach()
#if (NOT CUDAProfilingAPI_INCLUDE_DIR AND CUDAProfilingAPI_FIND_REQUIRED)
#    message("Couldn't find nvperf_host.h; tried:")
#    foreach(include_path IN LISTS TRY_DIRECTORIES)
#        message("  ${include_path}")
#    endforeach()
#    message(FATAL_ERROR)
#endif()

#foreach(lib_path IN LISTS CUPTI_LIB64_DIR CUDA_TARGETS_LIB64_DIR)
#    find_library(nvperf_host_LIBRARY
#            NAMES nvperf_host
#            HINTS ${lib_path})
#endforeach()
set(nvperf_host_LIBRARY_TRY ${CUPTI_LIB64_DIR} ${CUDA_TARGETS_LIB64_DIR})
TRY_FIND_LIB(nvperf_host_LIBRARY nvperf_host nvperf_host_LIBRARY_TRY)

foreach(lib_path IN LISTS CUPTI_LIB64_DIR CUDA_TARGETS_LIB64_DIR)
    find_library(nvperf_target_LIBRARY
            NAMES nvperf_target
            HINTS ${lib_path})
endforeach()

foreach(lib_path IN LISTS CUPTI_LIB64_DIR CUDA_TARGETS_LIB64_DIR)
    find_library(nvtx_LIBRARY
            NAMES nvToolsExt
            HINTS ${lib_path})
endforeach()

set(CUDAProfilingAPI_LIBRARY
        ${nvperf_host_LIBRARY}
        ${nvperf_target_LIBRARY}
        ${nvtx_LIBRARY}
        )
#message("CUDAProfilingAPI_LIBRARY = ${CUDAProfilingAPI_LIBRARY}")

add_library(nvperf_target SHARED IMPORTED)
set_target_properties(nvperf_target PROPERTIES
        IMPORTED_LOCATION ${nvperf_target_LIBRARY}
        # INTERFACE_COMPILE_DEFINITIONS "SOME_FEATURE"
        )

add_library(nvperf_host SHARED IMPORTED)
set_target_properties(nvperf_host PROPERTIES
        IMPORTED_LOCATION ${nvperf_host_LIBRARY}
        # INTERFACE_COMPILE_DEFINITIONS "SOME_FEATURE"
        )

add_library(nvtx SHARED IMPORTED)
set_target_properties(nvtx PROPERTIES
        IMPORTED_LOCATION ${nvtx_LIBRARY}
        # INTERFACE_COMPILE_DEFINITIONS "SOME_FEATURE"
        )

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(CUDAProfilingAPI_PROCESS_INCLUDES ${CUDAProfilingAPI_INCLUDE_DIR})
set(CUDAProfilingAPI_PROCESS_LIBS ${CUDAProfilingAPI_LIBRARY})
#message("CUDAProfilingAPI_PROCESS_INCLUDES = ${CUDAProfilingAPI_PROCESS_INCLUDES}")
#message("CUDAProfilingAPI_PROCESS_LIBS = ${CUDAProfilingAPI_PROCESS_LIBS}")
libfind_process(CUDAProfilingAPI)
#message("CUDAProfilingAPI_LIBRARIES = ${CUDAProfilingAPI_LIBRARIES}")
#message("CUDAProfilingAPI_INCLUDE_DIRS = ${CUDAProfilingAPI_INCLUDE_DIRS}")

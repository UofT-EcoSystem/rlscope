# - Try to find Libcupti
# Once done, this will define
#
#  Libcupti_FOUND - system has Libcupti
#  Libcupti_INCLUDE_DIRS - the Libcupti include directories
#  Libcupti_LIBRARIES - link these to use Libcupti

include(LibFindMacros)

# Dependencies
#libfind_package(Libcupti CUDA)
find_package(CUDA)

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(Libcupti_PKGCONF Libcupti)

# Include dir
find_path(Libcupti_INCLUDE_DIR
        NAMES cupti.h
        PATHS ${Libcupti_PKGCONF_INCLUDE_DIRS}
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include
        )

#PATHS ${Libcupti_PKGCONF_LIBRARY_DIRS}
# Finally the library itself
find_library(Libcupti_LIBRARY
        NAMES cupti
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64
        )

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(Libcupti_PROCESS_INCLUDES Libcupti_INCLUDE_DIR)
set(Libcupti_PROCESS_LIBS Libcupti_LIBRARY)
libfind_process(Libcupti)

# - Try to find MySpdlog
# Once done, this will define
#
#  MySpdlog_FOUND - system has MySpdlog
#  MySpdlog_INCLUDE_DIRS - the MySpdlog include directories
#  MySpdlog_LIBRARIES - link these to use MySpdlog

include(LibFindMacros)

# Dependencies
#libfind_package(CUDA)
#find_package(CUDA)

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(MySpdlog_PKGCONF MySpdlog)

find_library(MySpdlog_LIBRARY
        NAMES spdlog)
if (NOT MySpdlog_LIBRARY AND MySpdlog_FIND_REQUIRED)
    message("Couldn't find MySpdlog_LIBRARY")
    message(FATAL_ERROR)
endif()

find_path(MySpdlog_INCLUDE_DIR
        NAMES spdlog/spdlog.h)
if (NOT MySpdlog_INCLUDE_DIR AND MySpdlog_FIND_REQUIRED)
    message("Couldn't find MySpdlog/MySpdlog.h")
    message(FATAL_ERROR)
endif()

set(MySpdlog_PROCESS_INCLUDES ${MySpdlog_INCLUDE_DIR})
set(MySpdlog_PROCESS_LIBS ${MySpdlog_LIBRARY})
#message("MySpdlog_PROCESS_INCLUDES = ${MySpdlog_PROCESS_INCLUDES}")
#message("MySpdlog_PROCESS_LIBS = ${MySpdlog_PROCESS_LIBS}")
libfind_process(MySpdlog)
#message("MySpdlog_LIBRARIES = ${MySpdlog_LIBRARIES}")
#message("MySpdlog_INCLUDE_DIRS = ${MySpdlog_INCLUDE_DIRS}")

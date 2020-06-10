# Recursively gather source files in current directory.
macro(GlobProtobufSources VAR)
    #    message("GlobProtobufSources: VAR = ${VAR}")
    file(GLOB_RECURSE ${VAR}
            *.proto
            )
    #    message("GlobProtobufSources: ${VAR} = ${${VAR}}")
    #    message("${VAR} = ${${VAR}}")
endmacro()

# Recursively gather source files in current directory.
macro(GlobSources VAR)
    #    message("GlobSources: VAR = ${VAR}")
    file(GLOB_RECURSE ${VAR}
            *.cu
            *.cpp
            *.cc
            *.c)
    #    message("${VAR} = ${${VAR}}")
endmacro()

macro(_GatherTargetSources TARGET_VAR SOURCE_FILES_VAR)
    GlobSources(${SOURCE_FILES_VAR})
    GlobProtobufSources(PROTO_BUF_SOURCE_FILES)

    get_filename_component(DIRNAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    # message("DIRNAME = ${DIRNAME}")
    set(${TARGET_VAR} ${DIRNAME})
    # message("TARGET_VAR (${TARGET_VAR}) = ${${TARGET_VAR}}")

    list(LENGTH PROTO_BUF_SOURCE_FILES PROTO_BUF_SOURCE_FILES_LENGTH)
    if (PROTO_BUF_SOURCE_FILES_LENGTH GREATER 0)
        #        message("> PROTO_BUF_SOURCE_FILES_LENGTH = ${PROTO_BUF_SOURCE_FILES_LENGTH}")
        #        message("> PROTO_SRCS = ${PROTO_SRCS}")
        protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS
                ${PROTO_BUF_SOURCE_FILES})
        list(APPEND ${SOURCE_FILES_VAR} ${PROTO_SRCS})
    endif()
endmacro()

function(_AddTargetDependencies TARGET)
    # message("TARGET = ${TARGET}")
    AddCUDA(${TARGET})
    add_backward(${TARGET})

    find_package(MySpdlog REQUIRED)
    target_link_libraries(${TARGET} PRIVATE ${MySpdlog_LIBRARIES})
    target_include_directories(${TARGET} PRIVATE ${MySpdlog_INCLUDE_DIRS})
    # message("> MySpdlog_LIBRARIES = ${MySpdlog_LIBRARIES}")
    # message("> MySpdlog_INCLUDE_DIRS = ${MySpdlog_INCLUDE_DIRS}")

    target_link_libraries(${TARGET} PRIVATE nlohmann_json::nlohmann_json)

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        get_target_property(CUR_FLAG ${TARGET} COMPILE_DEFINITIONS)
        set_target_properties(${TARGET} PROPERTIES COMPILE_DEFINITIONS "${CUR_FLAG};SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG")
        get_target_property(CUR_FLAG ${TARGET} COMPILE_DEFINITIONS)
    else()
        get_target_property(CUR_FLAG ${TARGET} COMPILE_DEFINITIONS)
        set_target_properties(${TARGET} PROPERTIES COMPILE_DEFINITIONS "${CUR_FLAG};SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_OFF")
        get_target_property(CUR_FLAG ${TARGET} COMPILE_DEFINITIONS)
    endif()
endfunction()

# Build CUDA library from recursively discovered source files.
macro(MakeCudaLibraryFromDir LIBNAME_VAR SOURCE_FILES)
    _GatherTargetSources(${LIBNAME_VAR} ${SOURCE_FILES})
    #    message("SOURCE_FILES = ${SOURCE_FILES}")
    #    message("LIBNAME_VAR (${LIBNAME_VAR}) = ${${LIBNAME_VAR}}")
    cuda_add_library(${${LIBNAME_VAR}}
            ${${SOURCE_FILES}})
    _AddTargetDependencies(${${LIBNAME_VAR}})
    install(
            TARGETS ${${LIBNAME_VAR}}
            DESTINATION lib
            CONFIGURATIONS Debug)
    install(
            TARGETS ${${LIBNAME_VAR}}
            DESTINATION lib
            CONFIGURATIONS Release)
endmacro()

# Buld CUDA executable from recursively discovered source files.
macro(MakeCudaExecutableFromDir EXEC_NAME_VAR SOURCE_FILES)
    _GatherTargetSources(${EXEC_NAME_VAR} ${SOURCE_FILES})
    #    message("SOURCE_FILES = ${SOURCE_FILES}")
    #    message("EXEC_NAME_VAR (${EXEC_NAME_VAR}) = ${${EXEC_NAME_VAR}}")
    cuda_add_executable(${${EXEC_NAME_VAR}}
            ${${SOURCE_FILES}})
    _AddTargetDependencies(${${EXEC_NAME_VAR}})
    install(
            TARGETS ${${EXEC_NAME_VAR}}
            DESTINATION bin
            CONFIGURATIONS Debug)
    install(
            TARGETS ${${EXEC_NAME_VAR}}
            DESTINATION bin
            CONFIGURATIONS Release)
endmacro()

#find_package(CUDAProfilingAPI REQUIRED)
#include_directories(${CUDAProfilingAPI_INCLUDE_DIR})
# Gets included by libraries that require it (e.g., profilerhost_util)
#link_libraries(${CUDAProfilingAPI_LIBRARY})

# Include directories containing generated *.pb.h protobuf header files.
include_directories(
        ${CMAKE_CURRENT_BINARY_DIR}/src/libs
)

add_subdirectory(./src/libs)

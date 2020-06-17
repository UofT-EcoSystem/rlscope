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

macro(_GatherTargetSources TARGET_VAR SOURCE_FILES_VAR PROTO_SRCS_VAR PROTO_HDRS_VAR)
    GlobSources(${SOURCE_FILES_VAR})
    GlobProtobufSources(PROTO_BUF_SOURCE_FILES)

    get_filename_component(DIRNAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    # message("DIRNAME = ${DIRNAME}")
    set(${TARGET_VAR} ${DIRNAME})
    # message("TARGET_VAR (${TARGET_VAR}) = ${${TARGET_VAR}}")

    set(${PROTO_SRCS_VAR})
    set(${PROTO_HDRS_VAR})
    list(LENGTH PROTO_BUF_SOURCE_FILES PROTO_BUF_SOURCE_FILES_LENGTH)
    if (PROTO_BUF_SOURCE_FILES_LENGTH GREATER 0)
        # message("> PROTOBUF: ${${TARGET_VAR}}: PROTO_BUF_SOURCE_FILES_LENGTH = ${PROTO_BUF_SOURCE_FILES_LENGTH}")
#        message("> PROTOBUF: ${${TARGET_VAR}}: protobuf_generate_cpp")
#        message("            ${PROTO_SRCS_VAR}")
#        message("            ${PROTO_HDRS_VAR}")
#        message("            ${PROTO_BUF_SOURCE_FILES}")
        protobuf_generate_cpp(${PROTO_SRCS_VAR} ${PROTO_HDRS_VAR}
                ${PROTO_BUF_SOURCE_FILES})
#        include_directories(
#                ${CMAKE_CURRENT_BINARY_DIR}/src/libs
#        )
        # message("> PROTOBUF: ${${TARGET_VAR}}: ${PROTO_SRCS_VAR} = ${${PROTO_SRCS_VAR}}")
        # message("> PROTOBUF: ${${TARGET_VAR}}: ${PROTO_HDRS_VAR} = ${${PROTO_HDRS_VAR}}")
        # Glob for source files again in case pb.h / pb.cc files have been generated.
        # GlobSources(${SOURCE_FILES_VAR})
        list(APPEND ${SOURCE_FILES_VAR} ${${PROTO_SRCS_VAR}} ${${PROTO_HDRS_VAR}})
        # message("> PROTOBUF: ${${TARGET_VAR}}: append sources: ${${PROTO_SRCS_VAR}} ${${PROTO_HDRS_VAR}}")
    endif()

    RmGTestSources(${SOURCE_FILES_VAR})
endmacro()

function(_AddTargetDependencies TARGET)
    # message("TARGET = ${TARGET}")
    AddCUDA(${TARGET})
    add_backward(${TARGET})
    AddLoggingDeps(${TARGET})
    AddGFlagsDeps(${TARGET})

#    find_package(MySpdlog REQUIRED)
#    target_link_libraries(${TARGET} PRIVATE ${MySpdlog_LIBRARIES})
#    target_include_directories(${TARGET} PRIVATE ${MySpdlog_INCLUDE_DIRS})
#    # message("> MySpdlog_LIBRARIES = ${MySpdlog_LIBRARIES}")
#    # message("> MySpdlog_INCLUDE_DIRS = ${MySpdlog_INCLUDE_DIRS}")
#
#    target_link_libraries(${TARGET} PRIVATE nlohmann_json::nlohmann_json)
#
#    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#        get_target_property(CUR_FLAG ${TARGET} COMPILE_DEFINITIONS)
#        set_target_properties(${TARGET} PROPERTIES COMPILE_DEFINITIONS "${CUR_FLAG};SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG")
#        get_target_property(CUR_FLAG ${TARGET} COMPILE_DEFINITIONS)
#    else()
#        get_target_property(CUR_FLAG ${TARGET} COMPILE_DEFINITIONS)
#        set_target_properties(${TARGET} PROPERTIES COMPILE_DEFINITIONS "${CUR_FLAG};SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_OFF")
#        get_target_property(CUR_FLAG ${TARGET} COMPILE_DEFINITIONS)
#    endif()

endfunction()

function(_AddProtoDeps TARGET PROTO_SRCS PROTO_HDRS)
    # message("> _AddProtoDeps : PROTO_SRCS = ${PROTO_SRCS}")
    AddProtobufDeps(${TARGET})
    # Magical incantation to FORCE protobuf sources to be generated BEFORE people that depend
    # on range_sampling library (librlscope.so) are built.
    # OTHERWISE, first cmake build fails every time.
    #
    # Long story short: protobuf uses add_custom_command to output generated files, but cmake sucks
    # at letting you depend on those files directly, so we need to add an extra level of indirection
    # (add_custom_target) to depend on the generated proto srcs.
    add_custom_target(${TARGET}_cpp_proto_srcs
            DEPENDS ${PROTO_SRCS} ${PROTO_HDRS})
    add_dependencies(${TARGET} ${TARGET}_cpp_proto_srcs)
endfunction()

# Build CUDA library from recursively discovered source files.
macro(MakeCudaLibraryFromDir LIBNAME_VAR SOURCE_FILES)
    _GatherTargetSources(${LIBNAME_VAR} ${SOURCE_FILES} PROTO_SRCS PROTO_HDRS)
#    message("> PROTOBUF: ${${LIBNAME_VAR}}: SOURCE_FILES (${SOURCE_FILES}) = ${${SOURCE_FILES}}")
    # message("> PROTOBUF: ${${LIBNAME_VAR}}: LIBNAME_VAR (${LIBNAME_VAR}) = ${${LIBNAME_VAR}}")
    cuda_add_library(${${LIBNAME_VAR}}
            ${${SOURCE_FILES}}
            STATIC)
    _AddTargetDependencies(${${LIBNAME_VAR}})
    # Third party: protobuf
    if (PROTO_SRCS OR PROTO_HDRS)
        message("> PROTO_SRCS = ${PROTO_SRCS}")
        _AddProtoDeps(${${LIBNAME_VAR}} ${PROTO_SRCS} ${PROTO_HDRS})
#        AddProtobufDeps(${${LIBNAME_VAR}})
#        # Magical incantation to FORCE protobuf sources to be generated BEFORE people that depend
#        # on range_sampling library (librlscope.so) are built.
#        # OTHERWISE, first cmake build fails every time.
#        #
#        # Long story short: protobuf uses add_custom_command to output generated files, but cmake sucks
#        # at letting you depend on those files directly, so we need to add an extra level of indirection
#        # (add_custom_target) to depend on the generated proto srcs.
#        add_custom_target(${${LIBNAME_VAR}}_cpp_proto_srcs
#                DEPENDS ${PROTO_SRCS} ${PROTO_HDRS})
#        add_dependencies(${${LIBNAME_VAR}} ${${LIBNAME_VAR}}_cpp_proto_srcs)
    endif()
#    install(
#            TARGETS ${${LIBNAME_VAR}}
#            DESTINATION lib
#            CONFIGURATIONS Debug)
#    install(
#            TARGETS ${${LIBNAME_VAR}}
#            DESTINATION lib
#            CONFIGURATIONS Release)
endmacro()

# Buld CUDA executable from recursively discovered source files.
macro(MakeCudaExecutableFromDir EXEC_NAME_VAR SOURCE_FILES)
    set(PROTO_SRCS)
    set(PROTO_HDRS)
    _GatherTargetSources(${EXEC_NAME_VAR} ${SOURCE_FILES} PROTO_SRCS PROTO_HDRS)
    #    message("SOURCE_FILES = ${SOURCE_FILES}")
    #    message("EXEC_NAME_VAR (${EXEC_NAME_VAR}) = ${${EXEC_NAME_VAR}}")
    cuda_add_executable(${${EXEC_NAME_VAR}}
            ${${SOURCE_FILES}})
    _AddTargetDependencies(${${EXEC_NAME_VAR}})
    # Third party: protobuf
    if (PROTO_SRCS OR PROTO_HDRS)
        _AddProtoDeps(${${EXEC_NAME_VAR}} PROTO_SRCS PROTO_HDRS)
    endif()
#    if (PROTO_SRCS OR PROTO_HDRS)
#        AddProtobufDeps(${${LIBNAME_VAR}})
#    endif()
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
#message(" >> PROTOBUF: include_directories: ${CMAKE_CURRENT_BINARY_DIR}/src/libs")
include_directories(
        ${CMAKE_CURRENT_BINARY_DIR}/src/libs
)

add_subdirectory(./src/libs)

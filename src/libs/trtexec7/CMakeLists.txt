# If defined, enable collection of GPU hw counters in trtexec7.
#set(RLS_ENABLE_HW_COUNTERS OFF)
set(RLS_ENABLE_HW_COUNTERS ON)

MakeCudaExecutableFromDir(EXEC_NAME SOURCE_FILES)
if (RLS_ENABLE_HW_COUNTERS)
    message("> RLS_ENABLE_HW_COUNTERS=ON")
    target_compile_definitions(${EXEC_NAME} PRIVATE -DRLS_ENABLE_HW_COUNTERS)
#    add_definitions(-DRLS_ENABLE_HW_COUNTERS)
else()
    message("> RLS_ENABLE_HW_COUNTERS=OFF")
endif()
target_link_libraries(${EXEC_NAME} PRIVATE tensorrt_common7)
if (RLS_ENABLE_HW_COUNTERS)
    target_link_libraries(${EXEC_NAME} PRIVATE range_sampling)
endif()
target_link_libraries(${EXEC_NAME} PRIVATE common_util)
if (RLS_ENABLE_HW_COUNTERS)
    AddCUPTIDependencies(${EXEC_NAME})
endif()
#target_link_libraries(${EXEC_NAME} PRIVATE range_sampling)
#AddCUPTIDependencies(${EXEC_NAME})
#target_link_libraries(${EXEC_NAME} PRIVATE nvperf_host)
#target_link_libraries(${EXEC_NAME} PRIVATE nvperf_target)

#find_library(LIBNVTOOLSEXT libnvToolsExt.so PATHS ${CUDA_PATH}/lib64)
#target_link_libraries(${EXEC_NAME} ${LIBNVTOOLSEXT})
#target_link_libraries(${EXEC_NAME} PRIVATE nvtx)

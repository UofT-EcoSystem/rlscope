MakeCudaExecutableFromDir(EXEC_NAME SOURCE_FILES)
target_link_libraries(${EXEC_NAME} PRIVATE tensorrt_common6)
target_link_libraries(${EXEC_NAME} PRIVATE range_sampling)
target_link_libraries(${EXEC_NAME} PRIVATE common_util)
AddCUPTIDependencies(${EXEC_NAME})
#target_link_libraries(${EXEC_NAME} PRIVATE range_sampling)
#AddCUPTIDependencies(${EXEC_NAME})
#target_link_libraries(${EXEC_NAME} PRIVATE nvperf_host)
#target_link_libraries(${EXEC_NAME} PRIVATE nvperf_target)

#find_library(LIBNVTOOLSEXT libnvToolsExt.so PATHS ${CUDA_PATH}/lib64)
#target_link_libraries(${EXEC_NAME} ${LIBNVTOOLSEXT})
#target_link_libraries(${EXEC_NAME} PRIVATE nvtx)

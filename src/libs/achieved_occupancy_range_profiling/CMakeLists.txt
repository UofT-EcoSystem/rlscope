MakeCudaExecutableFromDir(EXEC_NAME SOURCE_FILES)
target_link_libraries(${EXEC_NAME} PRIVATE common_util)
target_link_libraries(${EXEC_NAME} PRIVATE profilerhost_util)
AddCUPTIDependencies(${EXEC_NAME})

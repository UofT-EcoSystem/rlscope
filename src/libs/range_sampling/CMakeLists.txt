MakeCudaLibraryFromDir(LIBNAME SOURCE_FILES)
target_link_libraries(${LIBNAME} common_util)
target_link_libraries(${LIBNAME} nvperf_host)
target_link_libraries(${LIBNAME} nvperf_target)

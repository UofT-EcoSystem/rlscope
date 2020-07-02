MakeCudaLibraryFromDir(LIBNAME SOURCE_FILES)
#target_link_libraries(${LIBNAME} PRIVATE common_util)
AddTensorRTDeps(${LIBNAME})

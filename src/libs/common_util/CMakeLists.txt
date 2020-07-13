MakeCudaLibraryFromDir(LIBNAME SOURCE_FILES)
target_link_libraries(${LIBNAME} PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(${LIBNAME} PRIVATE boost_filesystem)

find_package(CUDAProfilingAPI REQUIRED)
#target_link_libraries(${LIBNAME} PUBLIC ${CUDAProfilingAPI_LIBRARIES})
target_include_directories(${LIBNAME} PUBLIC ${CUDAProfilingAPI_INCLUDE_DIRS})

AddLoggingDeps(${LIBNAME})

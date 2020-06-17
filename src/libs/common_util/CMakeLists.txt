MakeCudaLibraryFromDir(LIBNAME SOURCE_FILES)
target_link_libraries(${LIBNAME} PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(${LIBNAME} PRIVATE boost_filesystem)

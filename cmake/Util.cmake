macro(TRY_ALIAS_LIB LIB_ALIAS LIB_VAR)
    if (${LIB_VAR})
        add_library(${LIB_ALIAS} SHARED IMPORTED)
        set_target_properties(${LIB_ALIAS} PROPERTIES
                IMPORTED_LOCATION ${${LIB_VAR}})
    endif()
endmacro()

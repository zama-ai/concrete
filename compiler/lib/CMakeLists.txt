add_subdirectory(Dialect)
add_subdirectory(Conversion)

# CAPI needed only for python bindings
if (ZAMALANG_BINDINGS_PYTHON_ENABLED)
    add_subdirectory(CAPI)
endif()

add_subdirectory(Dialect)
# CAPI needed only for python bindings
if (ZAMALANG_BINDINGS_PYTHON_ENABLED)
    add_subdirectory(CAPI)
endif()

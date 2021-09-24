add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Support)
add_subdirectory(Runtime)
add_subdirectory(Bindings)

# CAPI needed only for python bindings
if (ZAMALANG_BINDINGS_PYTHON_ENABLED)
    add_subdirectory(CAPI)
endif()

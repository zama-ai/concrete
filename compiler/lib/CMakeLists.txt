add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Transforms)
add_subdirectory(Support)
add_subdirectory(Runtime)
add_subdirectory(ClientLib)
add_subdirectory(Bindings)
add_subdirectory(TestLib)

# CAPI needed only for python bindings
if (CONCRETELANG_BINDINGS_PYTHON_ENABLED)
    add_subdirectory(CAPI)
endif()

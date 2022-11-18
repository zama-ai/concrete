# CAPI is mainly used by python and need to throw exceptions for proper handling of errors on the python-side
add_compile_options(-fexceptions)

add_subdirectory(Dialect)
add_subdirectory(Support)

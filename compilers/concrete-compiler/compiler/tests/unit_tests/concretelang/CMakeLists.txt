add_custom_target(ConcretelangUnitTests)

add_compile_options(-fexceptions)

add_subdirectory(ClientLib)
add_subdirectory(SDFG)
add_subdirectory(TestLib)
add_subdirectory(Encodings)
add_subdirectory(Dialect)
add_subdirectory(Runtime)

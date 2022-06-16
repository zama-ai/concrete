
add_executable(end_to_end_benchmark end_to_end_benchmark.cpp)
target_link_libraries(end_to_end_benchmark benchmark::benchmark ConcretelangSupport EndToEndFixture)
set_source_files_properties(end_to_end_benchmark.cpp PROPERTIES COMPILE_FLAGS "-fno-rtti")


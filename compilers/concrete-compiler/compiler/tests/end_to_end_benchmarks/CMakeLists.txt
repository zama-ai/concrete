add_executable(end_to_end_benchmark end_to_end_benchmark.cpp)
target_link_libraries(
  end_to_end_benchmark
  benchmark::benchmark
  ConcretelangSupport
  ConcretelangClientLib
  ConcretelangServerLib
  ConcretelangRuntime
  EndToEndFixture)
set_source_files_properties(end_to_end_benchmark.cpp PROPERTIES COMPILE_FLAGS "-fno-rtti")

add_executable(end_to_end_mlbench end_to_end_mlbench.cpp)
target_link_libraries(
  end_to_end_mlbench
  benchmark::benchmark
  ConcretelangSupport
  ConcretelangClientLib
  ConcretelangServerLib
  ConcretelangRuntime
  EndToEndFixture)
set_source_files_properties(end_to_end_mlbench.cpp PROPERTIES COMPILE_FLAGS "-fno-rtti -fsized-deallocation")

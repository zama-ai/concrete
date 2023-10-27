add_library(EndToEndFixture EndToEndFixture.cpp)

target_link_libraries(EndToEndFixture PRIVATE ConcretelangSupport)
set_source_files_properties(EndToEndFixture.cpp PROPERTIES COMPILE_FLAGS "-fno-rtti -fsized-deallocation")

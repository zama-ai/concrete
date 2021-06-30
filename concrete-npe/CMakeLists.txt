# concrete-npe/CMakeLists.txt

set(SOURCE tests/cpp/tests.cpp)

add_executable(tests ${SOURCE})
target_link_libraries (tests LINK_PUBLIC concrete_npe)

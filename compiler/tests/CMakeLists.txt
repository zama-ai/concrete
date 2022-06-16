if(CONCRETELANG_UNIT_TESTS OR CONCRETELANG_BENCHMARK)
    add_subdirectory(fixture)
endif()

if(CONCRETELANG_UNIT_TESTS)
    add_subdirectory(unittest)
    add_subdirectory(Support)
    add_subdirectory(TestLib)
endif()

if(CONCRETELANG_BENCHMARK)
    add_subdirectory(benchmarks)
endif()

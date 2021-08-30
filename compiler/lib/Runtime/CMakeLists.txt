add_library(ZamalangRuntime SHARED
    context.c
    wrappers.c
)

target_link_libraries(ZamalangRuntime Concrete pthread m dl)
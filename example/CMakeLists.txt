add_executable(example example.cpp)

target_link_libraries(example PRIVATE bruteforce)
target_link_libraries(example PRIVATE nnhnsw)

add_executable(example example.cpp)

target_include_directories(example PRIVATE .)

target_link_libraries(example PRIVATE bruteforce)
target_link_libraries(example PRIVATE ccvi)

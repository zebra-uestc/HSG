add_executable(dehnsw_example dehnsw_example.cpp)
target_include_directories(dehnsw_example PRIVATE .)
target_link_libraries(dehnsw_example PRIVATE dehnsw)

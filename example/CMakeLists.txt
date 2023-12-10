add_executable(nnhnsw_example nnhnsw_example.cpp)
target_include_directories(nnhnsw_example PRIVATE .)
target_link_libraries(nnhnsw_example PRIVATE nnhnsw)

add_executable(ccvi_example ccvi_example.cpp)
target_include_directories(ccvi_example PRIVATE .)
target_link_libraries(ccvi_example PRIVATE ccvi)

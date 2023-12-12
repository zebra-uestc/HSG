add_executable(nnhnsw_example nnhnsw_example.cpp)
target_include_directories(nnhnsw_example PRIVATE .)
target_link_libraries(nnhnsw_example PRIVATE nnhnsw)

add_executable(ccvi_example ccvi_example.cpp)
target_include_directories(ccvi_example PRIVATE .)
target_link_libraries(ccvi_example PRIVATE ccvi)

add_executable(ehnsw_example ehnsw_example.cpp)
target_include_directories(ehnsw_example PRIVATE .)
target_link_libraries(ehnsw_example PRIVATE ehnsw)

add_executable(dehnsw_example dehnsw_example.cpp)
target_include_directories(dehnsw_example PRIVATE .)
target_link_libraries(dehnsw_example PRIVATE dehnsw)

add_executable(dccvi_example dccvi_example.cpp)
target_include_directories(dccvi_example PRIVATE .)
target_link_libraries(dccvi_example PRIVATE dccvi)

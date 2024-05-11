add_executable(test test.cpp)
target_include_directories(test PRIVATE .)
target_include_directories(test PRIVATE ../source)
# target_link_libraries(test PRIVATE miluann)

# add_executable(hnsw hnsw.cpp)
# target_include_directories(hnsw PRIVATE .)
# target_include_directories(hnsw PRIVATE ../source)

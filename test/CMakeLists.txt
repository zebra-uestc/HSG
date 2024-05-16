add_executable(PT PT.cpp)
target_include_directories(PT PRIVATE .)
target_include_directories(PT PRIVATE ../source)
# target_link_libraries(PT PRIVATE HSG)

add_executable(debug debug.cpp)
target_include_directories(debug PRIVATE .)
target_include_directories(debug PRIVATE ../source)

# add_executable(hnsw hnsw.cpp)
# target_include_directories(hnsw PRIVATE .)
# target_include_directories(hnsw PRIVATE ../source)

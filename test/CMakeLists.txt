add_executable(PT EXCLUDE_FROM_ALL PT.cpp)
target_include_directories(PT PRIVATE .)
target_include_directories(PT PRIVATE ../source)

add_executable(debug debug.cpp)
target_include_directories(debug PRIVATE .)
target_include_directories(debug PRIVATE ../source)

add_executable(hnsw EXCLUDE_FROM_ALL hnsw.cpp)
target_include_directories(hnsw PRIVATE .)

add_executable(WRA EXCLUDE_FROM_ALL write_reference_answer.cpp)
target_include_directories(WRA PRIVATE .)

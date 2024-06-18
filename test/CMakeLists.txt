add_executable(performence EXCLUDE_FROM_ALL performence.cpp)
target_include_directories(performence PRIVATE .)
target_include_directories(performence PRIVATE ../source)

add_executable(debug debug.cpp)
target_include_directories(debug PRIVATE .)
target_include_directories(debug PRIVATE ../source)

add_executable(optimize EXCLUDE_FROM_ALL optimize.cpp)
target_include_directories(optimize PRIVATE .)
target_include_directories(optimize PRIVATE ../source)

add_executable(hnsw EXCLUDE_FROM_ALL hnsw.cpp)
target_include_directories(hnsw PRIVATE .)

add_executable(WRA EXCLUDE_FROM_ALL write_reference_answer.cpp)
target_include_directories(WRA PRIVATE .)

add_executable(delete EXCLUDE_FROM_ALL delete.cpp)
target_include_directories(delete PRIVATE .)
target_include_directories(delete PRIVATE ../source)

add_executable(NSG EXCLUDE_FROM_ALL NSG.cpp)
target_include_directories(NSG PRIVATE .)

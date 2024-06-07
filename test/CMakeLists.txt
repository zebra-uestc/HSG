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

add_executable(DB EXCLUDE_FROM_ALL delete_both.cpp
        NSG.cpp)
target_include_directories(DB PRIVATE .)
target_include_directories(DB PRIVATE ../source)

add_executable(DR EXCLUDE_FROM_ALL delete_relevant.cpp)
target_include_directories(DR PRIVATE .)
target_include_directories(DR PRIVATE ../source)

add_executable(DI EXCLUDE_FROM_ALL delete_irrelevant.cpp)
target_include_directories(DI PRIVATE .)
target_include_directories(DI PRIVATE ../source)

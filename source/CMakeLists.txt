add_library(nnhnsw STATIC)
set_target_properties(nnhnsw PROPERTIES LINKER_LANGUAGE CXX)
target_sources(nnhnsw
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/nnhnsw.h
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/distance.h
)
target_include_directories(nnhnsw
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
)

add_library(bruteforce STATIC)
set_target_properties(bruteforce PROPERTIES LINKER_LANGUAGE CXX)
target_sources(bruteforce
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/bruteforce.h
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/distance.h
)
target_include_directories(bruteforce
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
)

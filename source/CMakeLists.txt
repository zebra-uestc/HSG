add_library(ehnsw STATIC)
set_target_properties(ehnsw PROPERTIES LINKER_LANGUAGE CXX)
target_sources(ehnsw
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/ehnsw.h
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/distance.h
)
target_include_directories(ehnsw
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

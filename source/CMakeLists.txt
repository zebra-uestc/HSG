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

add_library(ccvi STATIC)
set_target_properties(ccvi PROPERTIES LINKER_LANGUAGE CXX)
target_sources(ccvi
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/ccvi.h
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/distance.h
)
target_include_directories(ccvi
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
)

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

add_library(dehnsw STATIC)
set_target_properties(dehnsw PROPERTIES LINKER_LANGUAGE CXX)
target_sources(dehnsw
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/dehnsw.h
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/distance.h
)
target_include_directories(dehnsw
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
)

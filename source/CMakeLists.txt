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

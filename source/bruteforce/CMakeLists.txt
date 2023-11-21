add_library(bruteforce STATIC)
set_target_properties(bruteforce PROPERTIES LINKER_LANGUAGE CXX)
target_sources(bruteforce
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/bruteforce.h
)
target_include_directories(bruteforce
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
)
target_link_libraries(bruteforce PUBLIC distance)

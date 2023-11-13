add_library(bruteforce STATIC)
target_sources(bruteforce
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/bruteforce.cpp
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/bruteforce.h
)
target_include_directories(bruteforce
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
)
target_link_libraries(bruteforce PUBLIC distance)

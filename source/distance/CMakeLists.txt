add_library(distance STATIC)
target_sources(distance
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/distance.cpp
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/distance.h
)
target_include_directories(distance
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
)

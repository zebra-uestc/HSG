add_library(distance STATIC)
set_target_properties(distance PROPERTIES LINKER_LANGUAGE CXX)
target_sources(distance
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/distance.h
)
target_include_directories(distance
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
)

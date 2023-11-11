add_library(nnhnsw STATIC)
target_sources(nnhnsw
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/data_struct.h
        ${CMAKE_CURRENT_LIST_DIR}/data_struct.cpp
        ${CMAKE_CURRENT_LIST_DIR}/nnhnsw.cpp
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/nnhnsw.h
)
target_include_directories(nnhnsw
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
)
target_link_libraries(nnhnsw distance)

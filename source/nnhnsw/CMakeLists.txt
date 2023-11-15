add_library(nnhnsw STATIC)
set_target_properties(nnhnsw PROPERTIES LINKER_LANGUAGE CXX)
target_sources(nnhnsw
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/data_struct.h
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/nnhnsw.h
)
target_include_directories(nnhnsw
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
)
target_link_libraries(nnhnsw PUBLIC distance)

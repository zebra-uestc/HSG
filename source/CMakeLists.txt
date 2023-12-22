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

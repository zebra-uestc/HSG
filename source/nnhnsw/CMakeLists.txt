add_library(nnhnsw "")
target_sources(nnhnsw
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/data_struct.hpp
    ${CMAKE_CURRENT_LIST_DIR}/data_struct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/nnhnsw.hpp
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/nnhnsw.hpp
  )
target_include_directories(nnhnsw
  PUBLIC
      ${CMAKE_CURRENT_LIST_DIR}
  )
target_link_libraries(nnhnsw distance)

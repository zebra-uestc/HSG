add_library(distance "")
target_sources(distance
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/distance.cpp
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/distance.hpp
  )
target_include_directories(distance
  PUBLIC
      ${CMAKE_CURRENT_LIST_DIR}
  )

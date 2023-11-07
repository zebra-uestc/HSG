cmake_minimum_required(VERSION 3.20)

project(
    vector-index
    VERSION 1.0.0.0
    DESCRIPTION "This project is uesd to test vector index algorithm. "
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-std=c++20 -fpic" CACHE STRING "" FORCE)

if(CMAKE_BUILD_TYPE STREQUAL "Release")

    set(CMAKE_CXX_FLAGS_RELEASE "$-DNDEBUG -O3 -flto" CACHE STRING "" FORCE)

    set(TARGET_PATH ${PROJECT_SOURCE_DIR}/build/release)

else()

    set(CMAKE_CXX_FLAGS_DEBUG "-g -gstabs -O0 -Wall -Wextra -Wpedantic" CACHE STRING "" FORCE)

    set(TARGET_PATH ${PROJECT_SOURCE_DIR}/build/debug)

endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")

cmake_host_system_information(RESULT PHYSICAL_CORES QUERY NUMBER_OF_PHYSICAL_CORES)
message(STATUS "Physical cores: ${PHYSICAL_CORES}")
cmake_host_system_information(RESULT LOGICAL_CORES QUERY NUMBER_OF_LOGICAL_CORES)
message(STATUS "Logical cores: ${LOGICAL_CORES}")

add_subdirectory(source)

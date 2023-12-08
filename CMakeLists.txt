cmake_minimum_required(VERSION 3.20)

project(
        vector-index
        VERSION 1.0.0.0
        DESCRIPTION "This project is used to test vector index algorithm. "
        LANGUAGES CXX
)

add_subdirectory(source)

if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)

    set(CMAKE_CXX_FLAGS "-std=c++20 -march=native" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -Ofast -flto -fpic" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall" CACHE STRING "" FORCE)

    message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
    message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
    message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
    message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")

    cmake_host_system_information(RESULT PHYSICAL_CORES QUERY NUMBER_OF_PHYSICAL_CORES)
    message(STATUS "Physical cores: ${PHYSICAL_CORES}")
    cmake_host_system_information(RESULT LOGICAL_CORES QUERY NUMBER_OF_LOGICAL_CORES)
    message(STATUS "Logical cores: ${LOGICAL_CORES}")

    add_subdirectory(example)

    find_package(pybind11 REQUIRED)
    pybind11_add_module(ehnswpy python_binding/python_binding.cpp)

endif ()

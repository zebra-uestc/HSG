cmake_minimum_required(VERSION 3.20)

project(
        miluann
        VERSION 1.0.0.0
        DESCRIPTION "Mi Lu ANN"
        LANGUAGES CXX
)

# if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CONFIGURATION_TYPES "Release;Debug" CACHE STRING "" FORCE)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "-std=c++20 -march=native -stdlib=libc++ -fuse-ld=lld -rtlib=compiler-rt -ferror-limit=0 -ftemplate-backtrace-limit=0" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -flto -fpic -fopenmp" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -pedantic-errors" CACHE STRING "" FORCE)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "-std=c++20 -march=native" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -O2 -flto -fpic" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic" CACHE STRING "" FORCE)
endif()

# 设置release模式下静态库文件目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/binary/release CACHE STRING "" FORCE)
# 设置release模式下动态库文件目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/binary/release CACHE STRING "" FORCE)
# 设置release模式下可执行文件目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/binary/release CACHE STRING "" FORCE)

# 设置debug模式下静态库文件目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/binary/debug CACHE STRING "" FORCE)
# 设置debug模式下动态库文件目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/binary/debug CACHE STRING "" FORCE)
# 设置debug模式下可执行文件目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/binary/debug CACHE STRING "" FORCE)

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")

cmake_host_system_information(RESULT PHYSICAL_CORES QUERY NUMBER_OF_PHYSICAL_CORES)
message(STATUS "Physical cores: ${PHYSICAL_CORES}")
cmake_host_system_information(RESULT LOGICAL_CORES QUERY NUMBER_OF_LOGICAL_CORES)
message(STATUS "Logical cores: ${LOGICAL_CORES}")

# add_subdirectory(source)
add_subdirectory(example)

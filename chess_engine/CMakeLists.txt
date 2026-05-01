cmake_minimum_required(VERSION 3.15)
project(AIChessbot CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Aggressive native optimization for Release.
if(MSVC)
    add_compile_options(/W4 /O2 /arch:AVX2)
else()
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
    add_compile_options($<$<CONFIG:Release>:-O3>)
    add_compile_options($<$<CONFIG:Release>:-march=native>)
    add_compile_options($<$<CONFIG:Release>:-DNDEBUG>)
endif()

file(GLOB SOURCES src/*.cpp)
add_executable(chess_engine ${SOURCES})

find_package(Threads REQUIRED)
target_link_libraries(chess_engine PRIVATE Threads::Threads)

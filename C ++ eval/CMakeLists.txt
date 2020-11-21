cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(demo)

find_package(Torch REQUIRED)

add_executable(example example.cpp)
target_link_libraries(example ${TORCH_LIBRARIES})
set_property(TARGET example PROPERTY CXX_STANDARD 11)

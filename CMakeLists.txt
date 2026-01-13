cmake_minimum_required(VERSION 3.10)
project(DivToMulti)
aux_source_directory(src sourceFiles)
add_executable(${PROJECT_NAME} ${sourceFiles})
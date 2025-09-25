cmake_minimum_required (VERSION 3.16)
project(tinyraytracer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets Gui)

if(Qt6_FOUND)
    message(STATUS "Qt6 found: ${Qt6_DIR}")
    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    qt_standard_project_setup()
endif()

find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

file(GLOB SOURCES src/*.h src/*.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE Qt6::Widgets Qt6::Gui)
set_target_properties(${PROJECT_NAME} PROPERTIES
    WIN32_EXECUTABLE TRUE
)


cmake_minimum_required(VERSION 3.10)
project(AttendanceSystem)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS}
                  ${PROJECT_SOURCE_DIR}/include)

# Add executable
add_executable(attendance_system src/main.cpp)

# Link libraries
target_link_libraries(attendance_system ${OpenCV_LIBS})

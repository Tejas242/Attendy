cmake_minimum_required(VERSION 3.10)
project(AttendanceSystem)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

# Add this to check if face module is available
if(NOT OpenCV_LIBS MATCHES "opencv_face")
    message(FATAL_ERROR "OpenCV face module not found. Please install opencv-contrib modules")
endif()

include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)

add_executable(attendance_system
    src/main.cpp
    src/FaceRecognitionSystem.cpp)

target_link_libraries(attendance_system ${OpenCV_LIBS})

#include "../include/FaceRecognitionSystem.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>

int main() {
  std::cout << "OpenCV version: " << CV_VERSION << std::endl;

  FaceRecognitionSystem system;

  if (!system.init()) {
    std::cout << "Failed to initialize system" << std::endl;
    return -1;
  }

  std::cout << "Adding People to the system..." << std::endl;
  system.addPerson("Person1");
  system.addPerson("Person2");

  system.train();

  system.startAttendanceSystem();

  return 0;
}

#include "../include/FaceRecognitionSystem.hpp"
#include "opencv2/opencv.hpp"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int main() {
  std::cout << "OpenCV version: " << CV_VERSION << std::endl;

  // Print current working directory
  std::cout << "Working directory: " << fs::current_path() << std::endl;

  FaceRecognitionSystem system;

  if (!system.init()) {
    std::cout << "Failed to initialize system" << std::endl;
    return -1;
  }

  system.showMenu();

  return 0;
}

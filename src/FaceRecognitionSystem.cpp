#include "../include/FaceRecognitionSystem.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <fstream>
#include <iostream>
#include <vector>

FaceRecognitionSystem::FaceRecognitionSystem() {
  recognizer = cv::face::LBPHFaceRecognizer::create();
}

FaceRecognitionSystem::~FaceRecognitionSystem() {
  if (cap.isOpened())
    cap.release();
}

bool FaceRecognitionSystem::init() {
  // Load face cascade classifier
  if (!face_cascade.load("../resources/haarcascade_frontalface_default.xml")) {
    std::cout << "Error: Could not load face cascade classifier." << std::endl;
    return false;
  }

  // init camera
  cap.open(0);
  if (!cap.isOpened()) {
    std::cout << "Error: Could not open camera." << std::endl;
    return false;
  }

  return true;
}

cv::Mat FaceRecognitionSystem::detectFace(const cv::Mat &frame) {
  cv::Mat gray;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

  std::vector<cv::Rect> faces;
  face_cascade.detectMultiScale(gray, faces, 1.1, 4);

  if (faces.empty())
    return cv::Mat();

  cv::Mat face = gray(faces[0]);

  cv::Mat resized_face;
  cv::resize(face, resized_face, cv::Size(100, 100));

  return resized_face;
}

void FaceRecognitionSystem::addPerson(const std::string &name) {
  std::cout << "Adding Person: " << name << std::endl;
  std::cout << "Capturing faces for training. Press 'c' to capture. Press 'q' "
               "when done"
            << std::endl;

  std::vector<cv::Mat> faces;
  cv::Mat frame, face;

  while (true) {
    cap.read(frame);
    if (frame.empty())
      continue;

    // show current frame
    cv::imshow("Adding Person", frame);

    char key = (char)cv::waitKey(1);
    if (key == 'q')
      break;

    if (key == 'c') {
      face = detectFace(frame);
      if (!face.empty()) {
        faces.push_back(face);
        cv::rectangle(frame, cv::Rect(0, 0, 100, 50), cv::Scalar(0, 0, 0), -1);
        cv::putText(frame, "Captured: " + std::to_string(faces.size()),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(255, 255, 255), 2);
        std::cout << "Face captured. Total faces: " << faces.size()
                  << std::endl;
      }
    }
  }

  cv::destroyWindow("Adding Person");

  // train
  if (!faces.empty()) {
    names.push_back(name);
    int label = names.size() - 1;

    // Add faces and labels to training data
    training_faces.insert(training_faces.end(), faces.begin(), faces.end());
    training_labels.insert(training_labels.end(), faces.size(), label);
  }
}

void FaceRecognitionSystem::train() {
  if (names.empty()) {
    std::cout << "No training data available" << std::endl;
    return;
  }

  try {
    // Train with collected data
    recognizer->train(training_faces, training_labels);
    std::cout << "Model trained with " << names.size() << " persons"
              << std::endl;
  } catch (const cv::Exception &e) {
    std::cout << "Training error: " << e.what() << std::endl;
  }
}

void FaceRecognitionSystem::saveAttendance() {
  auto now = std::chrono::system_clock::now();
  auto time = std::chrono::system_clock::to_time_t(now);

  std::ofstream file("attendance.csv", std::ios::app);

  for (const auto &entry : attendance) {
    if (entry.second) {
      file << entry.first << "," << std::ctime(&time);
    }
  }
}

void FaceRecognitionSystem::startAttendanceSystem() {
  if (names.empty()) {
    std::cout << "No training data available. Please add someone first."
              << std::endl;
    return;
  }

  cv::Mat frame, face;
  while (true) {
    cap.read(frame);
    if (frame.empty())
      continue;

    face = detectFace(frame);

    if (!face.empty()) {
      int label;
      double confidence;
      recognizer->predict(face, label, confidence);

      if (confidence < 100) {
        std::string name = names[label];
        attendance[name] = true;

        cv::putText(frame, name + " - Conf: " + std::to_string(confidence),
                    cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(0, 255, 0), 2);
      }
    }

    cv::imshow("Attendance System", frame);

    char key = (char)cv::waitKey(1);
    if (key == 'q')
      break;
  }

  saveAttendance();
}

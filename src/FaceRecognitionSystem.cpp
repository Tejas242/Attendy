#include "../include/FaceRecognitionSystem.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <fstream>
#include <iostream>
#include <opencv2/core/types.hpp>
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

void FaceRecognitionSystem::drawUI(cv::Mat &frame,
                                   const std::string &current_status) {
  cv::rectangle(frame, cv::Point(0, 0), cv::Point(frame.cols, 60),
                cv::Scalar(50, 50, 50), -1);

  // system status
  cv::putText(frame, "Status: " + current_status, cv::Point(10, 30),
              cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

  std::string people_info =
      "Registered People: " + std::to_string(names.size());
  cv::putText(frame, people_info, cv::Point(frame.cols - 300, 30),
              cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

  int history_y = 70;
  cv::putText(frame, "Recent Recognitions:", cv::Point(10, history_y),
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 200, 0), 2);

  history_y += 30;
  for (const auto &rec : recent_recognitions) {
    auto now = std::chrono::system_clock::now();
    auto seconds =
        std::chrono::duration_cast<std::chrono::seconds>(now - rec.timestamp)
            .count();

    std::string time_ago = std::to_string(seconds) + "s ago";
    std::string info = rec.name +
                       " (Conf: " + std::to_string(int(rec.confidence)) +
                       "%) " + time_ago;

    cv::putText(frame, info, cv::Point(10, history_y), cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(200, 200, 200), 1);
    history_y += 25;
  }
}

void FaceRecognitionSystem::addToHistory(const std::string &name,
                                         double confidence) {
  RecognitionHistory rec{name, confidence, std::chrono::system_clock::now()};

  recent_recognitions.insert(recent_recognitions.begin(), rec);
  if (recent_recognitions.size() > MAX_HISTORY) {
    recent_recognitions.pop_back();
  }
}

void FaceRecognitionSystem::startAttendanceSystem() {
  if (names.empty()) {
    std::cout << "No training data available. Please add someone first."
              << std::endl;
    return;
  }

  cv::Mat frame, face;
  std::string current_status = "Waiting for face...";

  while (true) {
    cap.read(frame);
    if (frame.empty())
      continue;

    // Create a copy for drawing
    cv::Mat display_frame = frame.clone();

    // Detect face
    std::vector<cv::Rect> faces;
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    face_cascade.detectMultiScale(gray, faces, 1.1, 4);

    // Process each detected face
    for (const auto &face_rect : faces) {
      // Draw rectangle around face
      cv::rectangle(display_frame, face_rect, cv::Scalar(0, 255, 0), 2);

      // Get and process the face
      cv::Mat face = gray(face_rect);
      cv::resize(face, face, cv::Size(100, 100));

      try {
        int label;
        double confidence;
        recognizer->predict(face, label, confidence);

        if (confidence < 100) {
          std::string name = names[label];
          attendance[name] = true;

          // Draw name and confidence
          std::string label_text =
              name + " (" + std::to_string(int(confidence)) + "%)";
          cv::Point text_pos(face_rect.x, face_rect.y - 10);
          cv::putText(display_frame, label_text, text_pos,
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

          current_status = "Recognized: " + name;
          addToHistory(name, confidence);
        } else {
          current_status = "Unknown face detected";
        }
      } catch (const cv::Exception &e) {
        current_status = "Recognition error";
      }
    }

    if (faces.empty()) {
      current_status = "Waiting for face...";
    }

    // Draw UI elements
    drawUI(display_frame, current_status);

    // Show the frame
    cv::imshow("Attendance System", display_frame);

    char key = (char)cv::waitKey(1);
    if (key == 'q')
      break;
  }

  saveAttendance();
}

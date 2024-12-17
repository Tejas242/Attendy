#include "../include/FaceRecognitionSystem.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core/types.hpp>
#include <vector>

namespace fs = std::filesystem;

FaceRecognitionSystem::FaceRecognitionSystem() {
  recognizer = cv::face::LBPHFaceRecognizer::create();
  createDirectories();
}

void FaceRecognitionSystem::createDirectories() {
  try {
    // Get absolute path
    fs::path current_path = fs::current_path();
    std::cout << "Current working directory: " << current_path << std::endl;

    fs::path data_path = current_path / ".." / "data";
    fs::path faces_path = data_path / "faces";

    std::cout << "Creating directory: " << data_path << std::endl;
    fs::create_directories(data_path);

    std::cout << "Creating directory: " << faces_path << std::endl;
    fs::create_directories(faces_path);
  } catch (const fs::filesystem_error &e) {
    std::cout << "Error creating directories: " << e.what() << std::endl;
  }
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

  bool loaded = loadTrainingData();
  std::cout << "Training data loaded: " << (loaded ? "Yes" : "No") << std::endl;
  std::cout << "Number of people: " << names.size() << std::endl;
  std::cout << "Number of faces: " << training_faces.size() << std::endl;
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
  std::cout << "\nAdding Person: " << name << std::endl;
  std::cout << "Instructions:" << std::endl;
  std::cout << "- Press 'c' to capture face" << std::endl;
  std::cout << "- Press 'r' to restart capture" << std::endl;
  std::cout << "- Press 's' to save and finish" << std::endl;
  std::cout << "- Press 'q' to quit without saving" << std::endl;
  std::cout << "\nAim for at least 5 different angles!" << std::endl;

  std::vector<cv::Mat> faces;
  cv::Mat frame, face;
  int required_faces = 5;

  while (true) {
    cap.read(frame);
    if (frame.empty())
      continue;

    // Draw guide box
    int center_x = frame.cols / 2;
    int center_y = frame.rows / 2;
    cv::rectangle(frame, cv::Point(center_x - 150, center_y - 150),
                  cv::Point(center_x + 150, center_y + 150),
                  cv::Scalar(255, 255, 255), 2);

    // Show progress
    std::string progress = "Captured: " + std::to_string(faces.size()) + "/" +
                           std::to_string(required_faces);
    cv::putText(frame, progress, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(255, 255, 255), 2);

    cv::imshow("Add Person", frame);

    char key = (char)cv::waitKey(1);

    if (key == 'c') {
      face = detectFace(frame);
      if (!face.empty()) {
        faces.push_back(face);
        std::cout << "Face captured. Total faces: " << faces.size()
                  << std::endl;
      }
    } else if (key == 'r') {
      faces.clear();
      std::cout << "Restarting capture..." << std::endl;
    } else if (key == 's' && faces.size() >= required_faces) {
      break;
    } else if (key == 'q') {
      cv::destroyWindow("Add Person");
      return;
    }
  }

  cv::destroyWindow("Add Person");

  if (!faces.empty()) {
    try {
      // Create absolute paths
      fs::path current_path = fs::current_path();
      fs::path person_dir = current_path / ".." / "data" / "faces" / name;

      std::cout << "Creating directory: " << person_dir << std::endl;
      fs::create_directories(person_dir);

      // Save faces
      for (size_t i = 0; i < faces.size(); i++) {
        fs::path face_path = person_dir / (std::to_string(i) + ".jpg");
        std::cout << "Saving face to: " << face_path << std::endl;

        bool success = cv::imwrite(face_path.string(), faces[i]);
        if (!success) {
          std::cout << "Failed to save face image: " << face_path << std::endl;
        }
      }

      // Update training data
      names.push_back(name);
      int label = names.size() - 1;
      training_faces.insert(training_faces.end(), faces.begin(), faces.end());
      training_labels.insert(training_labels.end(), faces.size(), label);

      // Save names
      fs::path names_path = current_path / ".." / "data" / "names.txt";
      std::cout << "Saving names to: " << names_path << std::endl;
      std::ofstream name_file(names_path, std::ios::app);
      name_file << name << std::endl;

      train();

      fs::path model_path = current_path / ".." / "data" / "model.yml";
      std::cout << "Saving model to: " << model_path << std::endl;
      recognizer->save(model_path.string());

      std::cout << "Person added successfully!" << std::endl;
    } catch (const std::exception &e) {
      std::cout << "Error while saving data: " << e.what() << std::endl;
    }
  }
}

void FaceRecognitionSystem::saveTrainingData() {
  try {
    recognizer->write(MODEL_PATH);

    std::ofstream name_file(NAMES_PATH);
    for (const auto &name : names) {
      name_file << name << std::endl;
    }
  } catch (const cv::Exception &e) {
    std::cout << "Error saving training data: " << e.what() << std::endl;
  }
}

bool FaceRecognitionSystem::loadTrainingData() {
  // clear prev data
  names.clear();
  training_faces.clear();
  training_labels.clear();

  bool hasData = false;

  if (fs::exists(NAMES_PATH)) {
    std::ifstream name_file(NAMES_PATH);
    std::string name;
    while (std::getline(name_file, name)) {
      if (!name.empty()) {
        names.push_back(name);
        hasData = true;
      }
    }
  }

  if (hasData) {
    int label = 0;
    for (const auto &name : names) {
      std::string person_dir = FACES_PATH + name + "/";
      if (fs::exists(person_dir)) {
        for (const auto &entry : fs::directory_iterator(person_dir)) {
          cv::Mat face = cv::imread(entry.path(), cv::IMREAD_GRAYSCALE);
          if (!face.empty()) {
            cv::resize(face, face, cv::Size(100, 100));
            training_faces.push_back(face);
            training_labels.push_back(label);
          }
        }
      }
      label++;
    }

    if (!training_faces.empty()) {
      train();
      return true;
    }
  }

  return false;
}

void FaceRecognitionSystem::listPeople() {
  std::cout << "\nRegistered People:" << std::endl;
  std::cout << "----------------" << std::endl;
  for (size_t i = 0; i < names.size(); i++) {
    std::cout << (i + 1) << ". " << names[i] << std::endl;
  }
  std::cout << "----------------" << std::endl;
}

void FaceRecognitionSystem::removePerson(const std::string &name) {
  auto it = std::find(names.begin(), names.end(), name);
  if (it != names.end()) {
    int label = it - names.begin();

    names.erase(it);                   // remove from vectors
    fs::remove_all(FACES_PATH + name); // remove face files

    // updates names file
    std::ofstream name_file(NAMES_PATH, std::ios::trunc);
    for (const auto &rem_name : names) {
      name_file << rem_name << std::endl;
    }
    name_file.close();

    training_faces.clear();
    training_labels.clear();

    // retrain
    int new_label = 0;
    for (const auto &person : names) {
      std::string person_dir = FACES_PATH + person + "/";
      if (fs::exists(person_dir)) {
        for (const auto &entry : fs::directory_iterator(person_dir)) {
          cv::Mat face = cv::imread(entry.path(), cv::IMREAD_GRAYSCALE);
          if (!face.empty()) {
            cv::resize(face, face, cv::Size(100, 100));
            training_faces.push_back(face);
            training_labels.push_back(new_label);
          }
        }
      }
      new_label++;
    }

    // Retrain if we have remaining data
    if (!training_faces.empty()) {
      train();
      saveTrainingData();
      std::cout << "Model retrained with remaining " << names.size()
                << " persons" << std::endl;
    } else {
      // If no faces left, remove the model file
      if (fs::exists(MODEL_PATH)) {
        fs::remove(MODEL_PATH);
      }
    }

    std::cout << "Person removed successfully: " << name << std::endl;
  } else {
    std::cout << "Person not found in database" << std::endl;
  }
}

void FaceRecognitionSystem::showMenu() {
  while (true) {
    std::cout << "\nAttendance System Menu" << std::endl;
    std::cout << "--------------------" << std::endl;
    std::cout << "1. Start Attendance System" << std::endl;
    std::cout << "2. Add New Person" << std::endl;
    std::cout << "3. Remove Person" << std::endl;
    std::cout << "4. List Registered People" << std::endl;
    std::cout << "5. Exit" << std::endl;
    std::cout << "Choice: ";

    int choice;
    std::cin >> choice;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    switch (choice) {
    case 1:
      startAttendanceSystem();
      break;
    case 2: {
      std::cout << "Enter name: ";
      std::string name;
      std::getline(std::cin, name);
      addPerson(name);
      break;
    }
    case 3: {
      listPeople();
      std::cout << "Enter name to remove: ";
      std::string name;
      std::getline(std::cin, name);
      removePerson(name);
      break;
    }
    case 4:
      listPeople();
      break;
    case 5:
      return;
    default:
      std::cout << "Invalid choice!" << std::endl;
    }
  }
}

void FaceRecognitionSystem::train() {
  std::cout << "Training faces count: " << training_faces.size() << std::endl;
  std::cout << "Training labels count: " << training_labels.size() << std::endl;

  if (training_faces.empty() || training_labels.empty()) {
    std::cout << "No training data available" << std::endl;
    return;
  }

  try {
    // Train with collected data
    recognizer->train(training_faces, training_labels);
    std::cout << "Model trained successfully with " << names.size()
              << " persons" << std::endl;
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
      // rectangle around face
      cv::rectangle(display_frame, face_rect, cv::Scalar(0, 255, 0), 2);

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

    drawUI(display_frame, current_status);

    cv::imshow("Attendance System", display_frame);

    char key = (char)cv::waitKey(1);
    if (key == 'q') {
      cv::destroyWindow("Attendance System");
      break;
    }
  }

  saveAttendance();
}

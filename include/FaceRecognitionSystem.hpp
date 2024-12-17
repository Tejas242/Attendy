#pragma once
#include <chrono>
#include <map>
#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class FaceRecognitionSystem {
private:
  cv::CascadeClassifier face_cascade;
  cv::Ptr<cv::face::LBPHFaceRecognizer> recognizer;
  cv::VideoCapture cap;

  std::vector<std::string> names;         // persons
  std::map<std::string, bool> attendance; // track attendace

  std::vector<cv::Mat> training_faces;
  std::vector<int> training_labels;

  // data paths
  const std::string DATA_PATH = "../data/";
  const std::string FACES_PATH = DATA_PATH + "faces/";
  const std::string MODEL_PATH = DATA_PATH + "model.yml";
  const std::string NAMES_PATH = DATA_PATH + "names.txt";

  struct RecognitionHistory {
    std::string name;
    double confidence;
    std::chrono::system_clock::time_point timestamp;
  };

  std::vector<RecognitionHistory> recent_recognitions;
  const size_t MAX_HISTORY = 5;

  cv::Mat detectFace(const cv::Mat &frame);
  void saveAttendance();
  void drawUI(cv::Mat &frame, const std::string &current_status);
  void addToHistory(const std::string &name, double confidence);

  bool loadTrainingData();
  void saveTrainingData();
  void createDirectories();

public:
  FaceRecognitionSystem();
  ~FaceRecognitionSystem();

  bool init();
  void addPerson(const std::string &name);
  void removePerson(const std::string &name);
  void listPeople();
  void train();
  void startAttendanceSystem();
  void showMenu();
};

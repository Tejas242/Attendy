#pragma once
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

  cv::Mat detectFace(const cv::Mat &frame);
  void saveAttendance();

public:
  FaceRecognitionSystem();
  ~FaceRecognitionSystem();

  bool init();
  void addPerson(const std::string &name);
  void train();
  void startAttendanceSystem();
};

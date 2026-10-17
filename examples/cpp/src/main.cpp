// Copyright (c) 2026 AutoVision contributors. Apache-2.0.
// Minimal C++ ONNX Runtime reference runtime for AutoVision exports.
//
// Build (see README): CMake with OpenCV + ONNX Runtime. The example is a
// reference, not a stable ABI/SDK. It assumes the exported graph's outputs
// are (logits, boxes) in that order and applies the same decode contract
// as <artifact>.postprocess.json (DETR top-k or PicoDet class-offset NMS).
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "postprocess.hpp"

namespace {

struct Options {
  std::string model;
  std::string image;
  int num_classes = 80;
  int top_k = 300;
  float threshold = 0.5f;
  int input_width = 640;
  int input_height = 640;
  std::string family = "detr";
  float nms_iou = 0.6f;
  bool class_agnostic = false;
};

void usage() {
  std::cerr
      << "usage: autovision_infer --model model.onnx --image demo.jpg "
         "--num-classes 80\n"
         "       [--top-k 300] [--threshold 0.5] [--input-width 640] "
         "[--input-height 640]\n"
         "       [--family detr|picodet] [--nms-iou 0.6] "
         "[--class-agnostic]\n";
}

bool parse(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    const std::string key = argv[index];
    auto next = [&](std::string* value) -> bool {
      if (index + 1 >= argc) {
        return false;
      }
      *value = argv[++index];
      return true;
    };
    if (key == "--model") {
      if (!next(&options->model)) return false;
    } else if (key == "--image") {
      if (!next(&options->image)) return false;
    } else if (key == "--num-classes") {
      options->num_classes = std::stoi(argv[++index]);
    } else if (key == "--top-k") {
      options->top_k = std::stoi(argv[++index]);
    } else if (key == "--threshold") {
      options->threshold = std::stof(argv[++index]);
    } else if (key == "--input-width") {
      options->input_width = std::stoi(argv[++index]);
    } else if (key == "--input-height") {
      options->input_height = std::stoi(argv[++index]);
    } else if (key == "--family") {
      if (!next(&options->family)) return false;
    } else if (key == "--nms-iou") {
      options->nms_iou = std::stof(argv[++index]);
    } else if (key == "--class-agnostic") {
      options->class_agnostic = true;
    } else {
      std::cerr << "unknown flag: " << key << "\n";
      return false;
    }
  }
  return !options->model.empty() && !options->image.empty();
}

std::vector<float> preprocess(const cv::Mat& bgr, int input_width,
                              int input_height) {
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(input_width, input_height));
  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
  const float mean[3] = {0.485f, 0.456f, 0.406f};
  const float std[3] = {0.229f, 0.224f, 0.225f};
  std::vector<float> tensor(static_cast<std::size_t>(3) * input_width * input_height);
  for (int channel = 0; channel < 3; ++channel) {
    for (int y = 0; y < input_height; ++y) {
      const float* row = rgb.ptr<float>(y);
      for (int x = 0; x < input_width; ++x) {
        const float value = row[x * 3 + channel];
        tensor[static_cast<std::size_t>(channel) * input_width * input_height +
               y * input_width + x] = (value - mean[channel]) / std[channel];
      }
    }
  }
  return tensor;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse(argc, argv, &options)) {
    usage();
    return 2;
  }
  cv::Mat image = cv::imread(options.image);
  if (image.empty()) {
    std::cerr << "cannot read image: " << options.image << "\n";
    return 2;
  }
  const int original_width = image.cols;
  const int original_height = image.rows;

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "autovision");
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  Ort::Session session(env, options.model.c_str(), session_options);
  Ort::AllocatorWithDefaultOptions allocator;

  const auto input_name = session.GetInputNameAllocated(0, allocator);
  const auto output0_name = session.GetOutputNameAllocated(0, allocator);
  const auto output1_name = session.GetOutputNameAllocated(1, allocator);
  const std::vector<const char*> input_names{input_name.get()};
  const std::vector<const char*> output_names{output0_name.get(),
                                              output1_name.get()};

  const auto input_shape =
      session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  int input_width = options.input_width;
  int input_height = options.input_height;
  if (input_shape.size() == 4 && input_shape[3] > 0) {
    input_width = static_cast<int>(input_shape[3]);
    input_height = static_cast<int>(input_shape[2]);
  }
  auto tensor = preprocess(image, input_width, input_height);
  const std::array<int64_t, 4> shape = {1, 3, input_height, input_width};
  auto memory = Ort::MemoryInfo::CreateCpu(
      OrtArenaAllocator, OrtMemTypeDefault);
  auto input_tensor = Ort::Value::CreateTensor<float>(
      memory, tensor.data(), tensor.size(), shape.data(), shape.size());
  std::vector<Ort::Value> outputs;
  try {
    outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(),
                          &input_tensor, 1, output_names.data(), 2);
  } catch (const std::exception& error) {
    std::cerr << "inference failed: " << error.what() << "\n";
    return 1;
  }

  const auto logits_shape =
      outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  const auto boxes_shape =
      outputs[1].GetTensorTypeAndShapeInfo().GetShape();
  if (logits_shape.size() != 3 || boxes_shape.size() != 3) {
    std::cerr << "expected logits (B,Q,C) and boxes (B,Q,4)\n";
    return 1;
  }
  const int batch = static_cast<int>(logits_shape[0]);
  const int queries = static_cast<int>(logits_shape[1]);
  const int classes = static_cast<int>(logits_shape[2]);
  if (classes != options.num_classes) {
    std::cerr << "warning: --num-classes " << options.num_classes
              << " != graph classes " << classes << "; using the graph\n";
  }

  autovision::DecodeConfig config;
  config.num_classes = classes;
  config.top_k = options.top_k;
  config.score_threshold = options.threshold;
  config.family = options.family == "picodet"
                      ? autovision::Family::Picodet
                      : autovision::Family::Detr;
  config.nms_iou = options.nms_iou;
  config.class_aware = !options.class_agnostic;
  config.max_detections = options.top_k;

  std::vector<autovision::ImageSize> sizes(
      static_cast<std::size_t>(batch),
      {original_width, original_height});
  auto detections = autovision::decode(
      outputs[0].GetTensorMutableData<float>(),
      outputs[1].GetTensorMutableData<float>(), batch, queries, sizes, config);

  for (int index = 0; index < batch; ++index) {
    for (const auto& detection : detections[index]) {
      std::cout << "{\"image\":" << index
                << ",\"label\":" << detection.label
                << ",\"score\":" << detection.score
                << ",\"box\":[" << detection.box.x1 << "," << detection.box.y1
                << "," << detection.box.x2 << "," << detection.box.y2
                << "]}\n";
    }
  }
  return 0;
}

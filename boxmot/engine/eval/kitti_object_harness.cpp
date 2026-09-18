// BoxMOT's standalone adapter. The unmodified KITTI evaluator is supplied by
// the user at installation; its server entry point is never called.
#define main boxmot_unused_kitti_server_main
#include "evaluate_object.cpp"
#undef main

#include <fstream>
#include <iomanip>
#include <memory>
#include <set>
#include <stdexcept>

namespace {
struct Result {
  int eligible = 0;
  double ap = 0;
  std::vector<double> precision;
};

void write_array(std::ostream &out, const std::vector<double> &values) {
  out << '[';
  for (size_t index = 0; index < values.size(); ++index) {
    if (index) out << ',';
    out << values[index];
  }
  out << ']';
}
}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--version") {
      std::cout << "boxmot-kitti-object-harness 1\n";
      return 0;
    }
    if (argc != 5)
      throw std::runtime_error("Expected GT_DIR PREDICTION_DIR FRAME_IDS_FILE OUTPUT_JSON");
    initGlobals();
    std::ifstream frame_list(argv[3]);
    if (!frame_list) throw std::runtime_error("Cannot open frame list");
    std::vector<std::vector<tGroundtruth>> ground_truth;
    std::vector<std::vector<tDetection>> predictions;
    std::set<std::string> seen;
    std::string frame;
    while (std::getline(frame_list, frame)) {
      if (frame.empty() || frame.find_first_not_of("0123456789") != std::string::npos ||
          !seen.insert(frame).second)
        throw std::runtime_error("Frame IDs must be distinct nonempty decimal strings");
      bool success = false;
      ground_truth.push_back(loadGroundtruth(std::string(argv[1]) + "/" + frame + ".txt", success));
      if (!success) throw std::runtime_error("Cannot read ground truth for frame " + frame);
      bool compute_aos = false;
      std::vector<bool> image(NUM_CLASS, false), ground(NUM_CLASS, false), spatial(NUM_CLASS, false);
      predictions.push_back(loadDetections(std::string(argv[2]) + "/" + frame + ".txt", compute_aos,
                                           image, ground, spatial, success));
      if (!success) throw std::runtime_error("Cannot read predictions for frame " + frame);
    }
    if (ground_truth.empty()) throw std::runtime_error("No frames selected");

    std::unique_ptr<FILE, decltype(&fclose)> stats(tmpfile(), fclose);
    if (!stats) throw std::runtime_error("Cannot create evaluator statistics file");
    Result results[2][2][3];
    for (int metric = 0; metric < 2; ++metric) {
      for (int cls = 0; cls < 2; ++cls) {
        for (int difficulty = 0; difficulty < 3; ++difficulty) {
          Result &result = results[metric][cls][difficulty];
          for (size_t index = 0; index < ground_truth.size(); ++index) {
            std::vector<int32_t> ignored_gt, ignored_det;
            std::vector<tGroundtruth> dontcare;
            cleanData(static_cast<CLASSES>(cls), ground_truth[index], predictions[index],
                      ignored_gt, dontcare, ignored_det, result.eligible,
                      static_cast<DIFFICULTY>(difficulty));
          }
          result.precision.assign(41, 0.0);
          if (!result.eligible) continue;
          std::vector<double> orientation;
          double (*overlap)(tDetection, tGroundtruth, int32_t) = box3DOverlap;
          if (metric == 0) overlap = imageBoxOverlap;
          const bool success = eval_class(
              stats.get(), nullptr, static_cast<CLASSES>(cls), ground_truth, predictions, false,
              overlap, result.precision, orientation,
              static_cast<DIFFICULTY>(difficulty), metric == 0 ? IMAGE : BOX3D);
          if (!success || result.precision.size() != 41)
            throw std::runtime_error("Official KITTI evaluator failed");
          for (size_t index = 0; index < result.precision.size(); ++index) {
            const double value = result.precision[index];
            if (!std::isfinite(value) || value < 0.0 || value > 1.0)
              throw std::runtime_error("Official KITTI evaluator returned invalid precision");
            if (index) result.ap += value * 100.0 / 40.0;
          }
        }
      }
    }

    std::ofstream out(argv[4]);
    if (!out) throw std::runtime_error("Cannot create evaluation output");
    out << std::setprecision(17);
    const char *metrics[] = {"2d", "3d"};
    const char *classes[] = {"car", "pedestrian"};
    const char *difficulties[] = {"easy", "moderate", "hard"};
    out << "{\"metrics\":{";
    for (int metric = 0; metric < 2; ++metric) {
      if (metric) out << ',';
      out << '"' << metrics[metric] << "\":{";
      for (int cls = 0; cls < 2; ++cls) {
        if (cls) out << ',';
        out << '"' << classes[cls] << "\":{";
        for (int difficulty = 0; difficulty < 3; ++difficulty) {
          if (difficulty) out << ',';
          out << '"' << difficulties[difficulty] << "\":";
          const Result &result = results[metric][cls][difficulty];
          if (result.eligible) out << result.ap;
          else out << "null";
        }
        out << '}';
      }
      out << '}';
    }
    out << "},\"curves\":{";
    for (int metric = 0; metric < 2; ++metric) {
      if (metric) out << ',';
      out << '"' << metrics[metric] << "\":{";
      for (int cls = 0; cls < 2; ++cls) {
        if (cls) out << ',';
        out << '"' << classes[cls] << "\":{";
        for (int difficulty = 0; difficulty < 3; ++difficulty) {
          if (difficulty) out << ',';
          const Result &result = results[metric][cls][difficulty];
          out << '"' << difficulties[difficulty] << "\":{\"eligible_ground_truth\":"
              << result.eligible << ",\"recall_samples\":[";
          for (int index = 0; index < 41; ++index) {
            if (index) out << ',';
            out << index / 40.0;
          }
          out << "],\"precision\":";
          write_array(out, result.precision);
          out << '}';
        }
        out << '}';
      }
      out << '}';
    }
    out << "}}\n";
    if (!out) throw std::runtime_error("Failed to write evaluation output");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "KITTI object evaluation: " << error.what() << '\n';
    return 2;
  }
}

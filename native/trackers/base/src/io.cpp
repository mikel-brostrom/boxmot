#include "boxmot/trackers/base/io.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace boxmot::trackers::base {

namespace {

std::array<cv::Point2f, 4> OrderCorners(const std::array<cv::Point2f, 4>& corners) {
    std::array<cv::Point2f, 4> ordered{};
    auto sum = [](const cv::Point2f& point) { return point.x + point.y; };
    auto diff = [](const cv::Point2f& point) { return point.y - point.x; };

    ordered[0] = *std::min_element(
        corners.begin(),
        corners.end(),
        [&](const cv::Point2f& lhs, const cv::Point2f& rhs) { return sum(lhs) < sum(rhs); }
    );
    ordered[2] = *std::max_element(
        corners.begin(),
        corners.end(),
        [&](const cv::Point2f& lhs, const cv::Point2f& rhs) { return sum(lhs) < sum(rhs); }
    );
    ordered[1] = *std::min_element(
        corners.begin(),
        corners.end(),
        [&](const cv::Point2f& lhs, const cv::Point2f& rhs) { return diff(lhs) < diff(rhs); }
    );
    ordered[3] = *std::max_element(
        corners.begin(),
        corners.end(),
        [&](const cv::Point2f& lhs, const cv::Point2f& rhs) { return diff(lhs) < diff(rhs); }
    );
    return ordered;
}

Eigen::MatrixXf LoadTextMatrix(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Failed to open text matrix: " + path.string());
    }

    std::vector<std::vector<float>> rows;
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        for (char& ch : line) {
            if (ch == ',') {
                ch = ' ';
            }
        }
        std::istringstream parser(line);
        std::vector<float> row;
        float value = 0.0F;
        while (parser >> value) {
            row.push_back(value);
        }
        if (!row.empty()) {
            rows.push_back(std::move(row));
        }
    }

    if (rows.empty()) {
        return Eigen::MatrixXf(0, 0);
    }

    const int cols = static_cast<int>(rows.front().size());
    Eigen::MatrixXf matrix(static_cast<int>(rows.size()), cols);
    for (int row = 0; row < static_cast<int>(rows.size()); ++row) {
        if (static_cast<int>(rows[static_cast<std::size_t>(row)].size()) != cols) {
            throw std::runtime_error("Inconsistent column count in text matrix: " + path.string());
        }
        for (int col = 0; col < cols; ++col) {
            matrix(row, col) = rows[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
        }
    }
    return matrix;
}

std::string ReadString(std::ifstream& stream, const std::size_t size) {
    std::string buffer(size, '\0');
    stream.read(buffer.data(), static_cast<std::streamsize>(size));
    if (!stream) {
        throw std::runtime_error("Failed to read npy header");
    }
    return buffer;
}

std::vector<int> ParseShape(const std::string& header) {
    const std::size_t start = header.find('(');
    const std::size_t end = header.find(')', start);
    if (start == std::string::npos || end == std::string::npos) {
        throw std::runtime_error("Failed to parse npy shape");
    }

    std::string shape_text = header.substr(start + 1, end - start - 1);
    std::vector<int> dims;
    std::stringstream shape_stream(shape_text);
    std::string token;
    while (std::getline(shape_stream, token, ',')) {
        if (token.find_first_not_of(" \t") == std::string::npos) {
            continue;
        }
        dims.push_back(std::stoi(token));
    }
    return dims;
}

Eigen::MatrixXf LoadNpyMatrix(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open npy matrix: " + path.string());
    }

    const std::string magic = ReadString(stream, 6);
    if (magic != "\x93NUMPY") {
        throw std::runtime_error("Invalid npy file: " + path.string());
    }

    const auto major = static_cast<unsigned char>(ReadString(stream, 1)[0]);
    const auto minor = static_cast<unsigned char>(ReadString(stream, 1)[0]);
    (void)minor;

    std::size_t header_len = 0;
    if (major == 1) {
        std::uint16_t len = 0;
        stream.read(reinterpret_cast<char*>(&len), sizeof(len));
        header_len = len;
    } else if (major == 2) {
        std::uint32_t len = 0;
        stream.read(reinterpret_cast<char*>(&len), sizeof(len));
        header_len = len;
    } else {
        throw std::runtime_error("Unsupported npy version in " + path.string());
    }

    const std::string header = ReadString(stream, header_len);
    if (header.find("False") == std::string::npos) {
        throw std::runtime_error("Fortran-order npy arrays are not supported: " + path.string());
    }

    const std::vector<int> dims = ParseShape(header);
    if (dims.empty()) {
        return Eigen::MatrixXf(0, 0);
    }

    const int rows = dims.size() >= 1 ? dims[0] : 0;
    const int cols = dims.size() >= 2 ? dims[1] : 1;
    if (rows == 0) {
        return Eigen::MatrixXf(0, cols);
    }

    const bool is_f8 =
        header.find("'descr': '<f8'") != std::string::npos || header.find("\"descr\": \"<f8\"") != std::string::npos;
    const bool is_f4 =
        header.find("'descr': '<f4'") != std::string::npos || header.find("\"descr\": \"<f4\"") != std::string::npos;
    if (!is_f4 && !is_f8) {
        throw std::runtime_error("Only float32/float64 npy matrices are supported: " + path.string());
    }

    Eigen::MatrixXf matrix(rows, cols);
    if (is_f4) {
        std::vector<float> buffer(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
        stream.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(float)));
        if (!stream) {
            throw std::runtime_error("Failed to read float32 npy payload: " + path.string());
        }
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                matrix(row, col) = buffer[static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(col)];
            }
        }
        return matrix;
    }

    std::vector<double> buffer(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
    stream.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(double)));
    if (!stream) {
        throw std::runtime_error("Failed to read float64 npy payload: " + path.string());
    }
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            matrix(row, col) = static_cast<float>(
                buffer[static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(col)]
            );
        }
    }
    return matrix;
}

}  // namespace

int RoundLikeNumpy(const double value) {
    return static_cast<int>(std::nearbyint(value));
}

Eigen::MatrixXf LoadNumericMatrix(const fs::path& path) {
    if (!path.empty() && path.extension() == ".npy") {
        return LoadNpyMatrix(path);
    }
    return LoadTextMatrix(path);
}

fs::path ResolveCacheFile(const fs::path& path_without_suffix) {
    const fs::path npy_path = path_without_suffix;
    if (fs::exists(npy_path)) {
        return npy_path;
    }
    fs::path txt_path = npy_path;
    txt_path.replace_extension(".txt");
    if (fs::exists(txt_path)) {
        return txt_path;
    }
    return {};
}

fs::path SequenceImageDir(const fs::path& seq_dir) {
    const fs::path img1 = seq_dir / "img1";
    return fs::exists(img1) ? img1 : seq_dir;
}

std::vector<fs::path> ListSequenceFrames(const fs::path& img_dir) {
    std::vector<fs::path> frames;
    if (!fs::exists(img_dir)) {
        return frames;
    }
    for (const auto& entry : fs::directory_iterator(img_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string ext = entry.path().extension().string();
        if (ext == ".jpg" || ext == ".png") {
            frames.push_back(entry.path());
        }
    }
    std::sort(frames.begin(), frames.end());
    return frames;
}

int ParseFrameId(const fs::path& path) {
    return std::stoi(path.stem().string());
}

int ReadSequenceFps(const fs::path& seq_dir) {
    std::ifstream stream(seq_dir / "seqinfo.ini");
    if (!stream) {
        return 0;
    }
    std::string line;
    while (std::getline(stream, line)) {
        if (line.rfind("frameRate=", 0) == 0) {
            return std::stoi(line.substr(std::string("frameRate=").size()));
        }
    }
    return 0;
}

std::unordered_set<int> ComputeWantedFrames(const std::vector<int>& frame_ids, const int orig_fps, const int target_fps) {
    std::unordered_set<int> wanted;
    if (frame_ids.empty() || orig_fps <= 0 || target_fps <= 0) {
        return wanted;
    }
    const int effective_target = std::min(orig_fps, target_fps);
    const float step = static_cast<float>(orig_fps) / static_cast<float>(effective_target);
    const int max_frame = *std::max_element(frame_ids.begin(), frame_ids.end());
    for (float value = 1.0F; value <= static_cast<float>(max_frame) + 1.0e-6F; value += step) {
        wanted.insert(static_cast<int>(value));
    }
    return wanted;
}

Eigen::MatrixXf FilterRowsByFrame(const Eigen::MatrixXf& matrix, const std::unordered_set<int>& keep_frames) {
    if (matrix.rows() == 0 || keep_frames.empty()) {
        return matrix;
    }
    std::vector<int> keep_indices;
    keep_indices.reserve(static_cast<std::size_t>(matrix.rows()));
    for (int row = 0; row < matrix.rows(); ++row) {
        if (keep_frames.count(static_cast<int>(matrix(row, 0))) > 0) {
            keep_indices.push_back(row);
        }
    }

    Eigen::MatrixXf filtered(static_cast<int>(keep_indices.size()), matrix.cols());
    for (int row = 0; row < static_cast<int>(keep_indices.size()); ++row) {
        filtered.row(row) = matrix.row(keep_indices[static_cast<std::size_t>(row)]);
    }
    return filtered;
}

LoadedDetectionSequence LoadDetectionSequence(
    const fs::path& mot_root,
    const fs::path& det_emb_root,
    const std::string& detector_name,
    const std::string& sequence_name,
    const int target_fps,
    const std::string_view tracker_name
) {
    const fs::path seq_dir = mot_root / sequence_name;
    const fs::path img_dir = SequenceImageDir(seq_dir);
    std::vector<fs::path> frame_paths = ListSequenceFrames(img_dir);
    if (frame_paths.empty()) {
        throw std::runtime_error("No frames found for sequence: " + sequence_name);
    }

    std::vector<int> frame_ids;
    frame_ids.reserve(frame_paths.size());
    for (const auto& path : frame_paths) {
        frame_ids.push_back(ParseFrameId(path));
    }

    const fs::path base_dir = det_emb_root / detector_name;
    const fs::path det_path = ResolveCacheFile(base_dir / "dets" / (sequence_name + ".npy"));
    if (det_path.empty()) {
        throw std::runtime_error("Missing detection cache for sequence: " + sequence_name);
    }

    Eigen::MatrixXf detections = LoadNumericMatrix(det_path);
    const int cols = detections.rows() == 0 ? 0 : detections.cols();
    if (cols != 0 && cols != 7 && cols != 8) {
        throw std::runtime_error(
            "Native " + std::string(tracker_name) + " supports AABB caches with 7 cols or OBB caches with 8 cols only."
        );
    }

    std::unordered_set<int> keep_frames;
    if (target_fps > 0 && detections.rows() > 0) {
        const int orig_fps = ReadSequenceFps(seq_dir);
        if (orig_fps > 0) {
            keep_frames = ComputeWantedFrames(frame_ids, orig_fps, target_fps);
            detections = FilterRowsByFrame(detections, keep_frames);
            FilterFrames(keep_frames, frame_ids, frame_paths);
        }
    }

    LoadedDetectionSequence sequence;
    sequence.name = sequence_name;
    sequence.detections = std::move(detections);
    sequence.frame_ids = std::move(frame_ids);
    sequence.frame_paths = std::move(frame_paths);
    sequence.keep_frames = std::move(keep_frames);
    sequence.is_obb = cols == 8;
    return sequence;
}

std::array<cv::Point2f, 4> CanonicalObbCorners(const Eigen::Matrix<double, 5, 1>& box) {
    const float cx = static_cast<float>(box[0]);
    const float cy = static_cast<float>(box[1]);
    const float width = static_cast<float>(std::max(box[2], 1.0e-4));
    const float height = static_cast<float>(std::max(box[3], 1.0e-4));
    const float angle = static_cast<float>(box[4]);
    const float c = std::cos(angle);
    const float s = std::sin(angle);

    const std::array<cv::Point2f, 4> rect = {
        cv::Point2f(-width / 2.0F, -height / 2.0F),
        cv::Point2f(width / 2.0F, -height / 2.0F),
        cv::Point2f(width / 2.0F, height / 2.0F),
        cv::Point2f(-width / 2.0F, height / 2.0F),
    };

    std::array<cv::Point2f, 4> corners{};
    for (std::size_t index = 0; index < rect.size(); ++index) {
        const float x = rect[index].x;
        const float y = rect[index].y;
        corners[index] = cv::Point2f((x * c) - (y * s) + cx, (x * s) + (y * c) + cy);
    }
    return OrderCorners(corners);
}

void FilterFrames(
    const std::unordered_set<int>& keep_frames,
    std::vector<int>& frame_ids,
    std::vector<fs::path>& frame_paths
) {
    if (keep_frames.empty()) {
        return;
    }

    std::vector<int> filtered_ids;
    std::vector<fs::path> filtered_paths;
    filtered_ids.reserve(frame_ids.size());
    filtered_paths.reserve(frame_paths.size());
    for (std::size_t index = 0; index < frame_ids.size(); ++index) {
        if (keep_frames.count(frame_ids[index]) > 0) {
            filtered_ids.push_back(frame_ids[index]);
            filtered_paths.push_back(frame_paths[index]);
        }
    }
    frame_ids = std::move(filtered_ids);
    frame_paths = std::move(filtered_paths);
}

}  // namespace boxmot::trackers::base

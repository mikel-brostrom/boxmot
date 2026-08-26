#include "boxmot/trackers/base/io.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace boxmot::trackers::base {

namespace {

std::array<cv::Point2f, 4> OrderCorners(const std::array<cv::Point2f, 4>& corners) {
    std::array<cv::Point2f, 4> ordered = corners;
    cv::Point2f center(0.0F, 0.0F);
    for (const auto& corner : corners) {
        center += corner;
    }
    center *= 0.25F;
    std::stable_sort(
        ordered.begin(),
        ordered.end(),
        [&](const cv::Point2f& lhs, const cv::Point2f& rhs) {
            return std::atan2(lhs.y - center.y, lhs.x - center.x) <
                std::atan2(rhs.y - center.y, rhs.x - center.x);
        });

    float twice_area = 0.0F;
    for (std::size_t index = 0; index < ordered.size(); ++index) {
        const auto& current = ordered[index];
        const auto& next = ordered[(index + 1U) % ordered.size()];
        twice_area += (current.x * next.y) - (current.y * next.x);
    }
    if (twice_area < 0.0F) {
        std::reverse(ordered.begin(), ordered.end());
    }

    const auto start = std::min_element(
        ordered.begin(),
        ordered.end(),
        [](const cv::Point2f& lhs, const cv::Point2f& rhs) {
            return lhs.y < rhs.y || (lhs.y == rhs.y && lhs.x < rhs.x);
        });
    std::rotate(ordered.begin(), start, ordered.end());
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

struct NpyMetadata {
    std::string header;
    std::vector<int> dims;
};

NpyMetadata ReadNpyMetadata(std::ifstream& stream, const fs::path& path) {
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
    if (!stream) {
        throw std::runtime_error("Failed to read npy header length: " + path.string());
    }

    std::string header = ReadString(stream, header_len);
    if (header.find("False") == std::string::npos) {
        throw std::runtime_error("Fortran-order npy arrays are not supported: " + path.string());
    }
    return {header, ParseShape(header)};
}

Eigen::MatrixXf LoadNpyMatrix(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open npy matrix: " + path.string());
    }

    const NpyMetadata metadata = ReadNpyMetadata(stream, path);
    const std::string& header = metadata.header;
    const std::vector<int>& dims = metadata.dims;
    if (dims.size() != 2U) {
        throw std::runtime_error("Native npy matrix must be 2D: " + path.string());
    }

    const int rows = dims[0];
    const int cols = dims[1];
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

cv::Mat LoadNpyImage(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open npy image: " + path.string());
    }

    const NpyMetadata metadata = ReadNpyMetadata(stream, path);
    const bool is_u1 =
        metadata.header.find("'descr': '|u1'") != std::string::npos ||
        metadata.header.find("\"descr\": \"|u1\"") != std::string::npos ||
        metadata.header.find("'descr': '<u1'") != std::string::npos ||
        metadata.header.find("\"descr\": \"<u1\"") != std::string::npos;
    const bool grayscale_2d = metadata.dims.size() == 2U;
    if (!is_u1 || (!grayscale_2d && metadata.dims.size() != 3U)) {
        throw std::runtime_error(
            "Native replay expects a C-order uint8 grayscale or HWC npy image: " + path.string()
        );
    }

    const int height = metadata.dims[0];
    const int width = metadata.dims[1];
    const int channels = grayscale_2d ? 1 : metadata.dims[2];
    if (height <= 0 || width <= 0 || channels <= 0) {
        throw std::runtime_error("Npy image dimensions must be positive: " + path.string());
    }

    const std::size_t pixel_count = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    if (
        pixel_count > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(channels)
    ) {
        throw std::runtime_error("Npy image is too large: " + path.string());
    }
    std::vector<std::uint8_t> source(pixel_count * static_cast<std::size_t>(channels));
    stream.read(reinterpret_cast<char*>(source.data()), static_cast<std::streamsize>(source.size()));
    if (!stream) {
        throw std::runtime_error("Failed to read uint8 npy image payload: " + path.string());
    }

    if (channels == 1) {
        if (grayscale_2d) {
            // Match MOTSequence.__iter__'s cv2.COLOR_GRAY2BGR conversion.
            cv::Mat image(height, width, CV_8UC3);
            auto* destination = image.ptr<std::uint8_t>();
            for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
                const std::uint8_t value = source[pixel];
                destination[pixel * 3U] = value;
                destination[pixel * 3U + 1U] = value;
                destination[pixel * 3U + 2U] = value;
            }
            return image;
        }
        cv::Mat image(height, width, CV_8UC1);
        std::memcpy(image.data, source.data(), source.size());
        return image;
    }
    if (channels < 3) {
        throw std::runtime_error("Npy image must have one or at least three channels: " + path.string());
    }

    // Match MOTSequence.__iter__, the Python tracking replay path: truncate
    // every multi-channel npy frame to its first three channels.
    const std::array<int, 3> indices{0, 1, 2};
    cv::Mat image(height, width, CV_8UC3);
    auto* destination = image.ptr<std::uint8_t>();
    for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
        const std::size_t source_offset = pixel * static_cast<std::size_t>(channels);
        const std::size_t destination_offset = pixel * 3U;
        destination[destination_offset] = source[source_offset + static_cast<std::size_t>(indices[0])];
        destination[destination_offset + 1U] = source[source_offset + static_cast<std::size_t>(indices[1])];
        destination[destination_offset + 2U] = source[source_offset + static_cast<std::size_t>(indices[2])];
    }
    return image;
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
        if (entry.path().filename().string().rfind("._", 0) == 0) {
            continue;
        }
        const std::string ext = entry.path().extension().string();
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".npy") {
            frames.push_back(entry.path());
        }
    }
    std::sort(frames.begin(), frames.end());

    std::unordered_map<std::string, fs::path> path_by_stem;
    for (const auto& frame : frames) {
        const std::string stem = frame.stem().string();
        const auto [existing, inserted] = path_by_stem.emplace(stem, frame);
        if (!inserted) {
            throw std::runtime_error(
                "Multiple image files found for frame stem '" + stem + "' in " +
                img_dir.string() + ": " + existing->second.filename().string() + ", " +
                frame.filename().string() +
                ". Keep exactly one of .jpg, .jpeg, .png, or .npy per frame."
            );
        }
    }
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
    const double step = static_cast<double>(orig_fps) / static_cast<double>(effective_target);
    const int max_frame = *std::max_element(frame_ids.begin(), frame_ids.end());
    const double stop = static_cast<double>(max_frame) + 1.0;
    for (std::size_t index = 0;; ++index) {
        // NumPy arange computes start + index * step in float64. Repeated
        // float32 accumulation changes integer truncation at boundary values.
        const double value = 1.0 + static_cast<double>(index) * step;
        if (value >= stop) {
            break;
        }
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
    const int source_detection_rows = static_cast<int>(detections.rows());
    // NumPy preserves the trailing dimension for empty arrays. Keep it so an
    // empty (0, 8) cache still declares OBB mode rather than becoming unknown.
    const int cols = static_cast<int>(detections.cols());
    if (cols != 0 && cols != 7 && cols != 8) {
        throw std::runtime_error(
            "Native " + std::string(tracker_name) + " supports AABB caches with 7 cols or OBB caches with 8 cols only."
        );
    }

    std::unordered_set<int> keep_frames;
    std::vector<int> retained_detection_rows;
    retained_detection_rows.reserve(static_cast<std::size_t>(source_detection_rows));
    for (int row = 0; row < source_detection_rows; ++row) {
        retained_detection_rows.push_back(row);
    }
    if (target_fps > 0) {
        const int orig_fps = ReadSequenceFps(seq_dir);
        if (orig_fps > 0) {
            keep_frames = ComputeWantedFrames(frame_ids, orig_fps, target_fps);
            retained_detection_rows.clear();
            for (int row = 0; row < detections.rows(); ++row) {
                if (keep_frames.count(static_cast<int>(detections(row, 0))) > 0) {
                    retained_detection_rows.push_back(row);
                }
            }
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
    sequence.retained_detection_rows = std::move(retained_detection_rows);
    sequence.source_detection_rows = source_detection_rows;
    sequence.is_obb = cols == 8;
    return sequence;
}

Eigen::MatrixXf LoadEmbeddingsCache(
    const fs::path& det_emb_root,
    const std::string& detector_name,
    const std::string& reid_name,
    const std::string& reid_preprocess,
    const std::string& sequence_name,
    const std::vector<int>& retained_detection_rows,
    const int source_detection_rows,
    const bool can_infer_embeddings
) {
    const fs::path base_dir = det_emb_root / detector_name;
    const fs::path emb_path = ResolveCacheFile(
        base_dir / "embs" / reid_name / reid_preprocess / (sequence_name + ".npy")
    );
    if (emb_path.empty()) {
        if (!can_infer_embeddings) {
            throw std::runtime_error("Missing embedding cache for sequence: " + sequence_name);
        }
        return Eigen::MatrixXf(0, 0);
    }

    Eigen::MatrixXf embeddings = LoadNumericMatrix(emb_path);
    if (embeddings.rows() != source_detection_rows) {
        if (embeddings.rows() == 0 && source_detection_rows > 0 && can_infer_embeddings) {
            return embeddings;
        }
        throw std::runtime_error(
            "Detection and embedding row counts do not match for sequence: " + sequence_name
        );
    }
    if (embeddings.rows() == 0) {
        return embeddings;
    }

    Eigen::MatrixXf aligned(
        static_cast<int>(retained_detection_rows.size()),
        embeddings.cols()
    );
    for (int row = 0; row < static_cast<int>(retained_detection_rows.size()); ++row) {
        const int source_row = retained_detection_rows[static_cast<std::size_t>(row)];
        if (source_row < 0 || source_row >= embeddings.rows()) {
            throw std::runtime_error(
                "Detection/embedding FPS row alignment is invalid for sequence: " + sequence_name
            );
        }
        aligned.row(row) = embeddings.row(source_row);
    }
    return aligned;
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

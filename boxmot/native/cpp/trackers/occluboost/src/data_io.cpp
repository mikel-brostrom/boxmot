#include "occluboost/data_io.hpp"

#include "boxmot/trackers/base/io.hpp"

#include <opencv2/imgcodecs.hpp>

#include <stdexcept>
#include <utility>

namespace occluboost {

LoadedSequence LoadSequence(const ReplayOptions& options) {
    boxmot::trackers::base::LoadedDetectionSequence base_sequence =
        boxmot::trackers::base::LoadDetectionSequence(
            options.mot_root,
            options.det_emb_root,
            options.detector_name,
            options.sequence,
            options.target_fps,
            "OccluBoost"
        );

    const bool can_infer_embeddings =
        !options.reid_model_path.empty() && options.reid_model_path.extension() == ".onnx";
    Eigen::MatrixXf embeddings = boxmot::trackers::base::LoadEmbeddingsCache(
        options.det_emb_root,
        options.detector_name,
        options.reid_name,
        options.reid_preprocess,
        options.sequence,
        base_sequence.keep_frames,
        static_cast<int>(base_sequence.detections.rows()),
        can_infer_embeddings
    );

    LoadedSequence sequence;
    sequence.name = std::move(base_sequence.name);
    sequence.detections = std::move(base_sequence.detections);
    sequence.embeddings = std::move(embeddings);
    sequence.is_obb = base_sequence.is_obb;
    sequence.frame_ids = std::move(base_sequence.frame_ids);
    sequence.frame_paths = std::move(base_sequence.frame_paths);
    return sequence;
}

std::vector<Detection> SliceDetectionsForFrame(
    const LoadedSequence& sequence,
    const int frame_id,
    std::size_t& row_offset,
    const float conf_threshold
) {
    return boxmot::trackers::base::SliceReplayDetectionsForFrame<Detection>(
        sequence.detections,
        sequence.is_obb,
        frame_id,
        row_offset,
        conf_threshold,
        [&](Detection& detection, const int row) {
            if (sequence.embeddings.rows() > row) {
                detection.embedding = sequence.embeddings.row(row).transpose();
            }
        }
    );
}

cv::Mat ReadImage(const fs::path& path) {
    return cv::imread(path.string(), cv::IMREAD_COLOR);
}

void WriteMotLine(std::ostream& stream, const int frame_id, const TrackOutput& track) {
    boxmot::trackers::base::WriteMotResultLine(stream, frame_id, track);
}

}  // namespace occluboost

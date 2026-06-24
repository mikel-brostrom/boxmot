#include "ocsort/data_io.hpp"

#include "boxmot/trackers/base/io.hpp"

#include <opencv2/imgcodecs.hpp>

#include <utility>

namespace ocsort {

LoadedSequence LoadSequence(const ReplayOptions& options) {
    boxmot::trackers::base::LoadedDetectionSequence base_sequence = boxmot::trackers::base::LoadDetectionSequence(
        options.mot_root,
        options.det_emb_root,
        options.detector_name,
        options.sequence,
        options.target_fps,
        "OCSORT"
    );

    LoadedSequence sequence;
    sequence.name = std::move(base_sequence.name);
    sequence.detections = std::move(base_sequence.detections);
    sequence.frame_ids = std::move(base_sequence.frame_ids);
    sequence.frame_paths = std::move(base_sequence.frame_paths);
    sequence.is_obb = base_sequence.is_obb;
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
        conf_threshold
    );
}

cv::Mat ReadImage(const fs::path& path) {
    return cv::imread(path.string(), cv::IMREAD_COLOR);
}

void WriteResultLine(std::ostream& stream, const int frame_id, const TrackOutput& track) {
    boxmot::trackers::base::WriteMotResultLine(stream, frame_id, track);
}

}  // namespace ocsort

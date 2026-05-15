#pragma once
// BoxMOT DeepStream adapter — NvDsTracker API context.
//
// This is the heart of the adapter: a multi-stream tracker context that
// maintains per-stream BoxMOT tracker instances and a shared TensorRT ReID
// model (matching DeepStream's batched ReID inference pattern).

#include "deepstream/adapter_types.hpp"
#include "deepstream/reid_tensorrt.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// Forward-declare the tracker types so we don't pull all headers here.
namespace botsort { class BotSortTracker; struct Config; struct Detection; struct TrackOutput; }
namespace bytetrack { class ByteTracker; }
namespace ocsort { class OcSortTracker; }
namespace sfsort { class SfSortTracker; }

namespace boxmot::deepstream {

// Opaque tracker handle per stream. Wraps whichever BoxMOT tracker is selected.
class PerStreamTracker {
public:
    virtual ~PerStreamTracker() = default;

    // Process detections for one frame. Returns tracked objects.
    virtual std::vector<TrackedObject> Update(
        const std::vector<cv::Rect>& det_boxes,
        const std::vector<float>& det_confs,
        const std::vector<int>& det_classes,
        const std::vector<std::vector<float>>& det_embeddings,
        const cv::Mat& frame
    ) = 0;

    virtual void Reset() = 0;
};

// Factory for per-stream trackers
std::unique_ptr<PerStreamTracker> CreatePerStreamTracker(
    const BoxMOTTrackerConfig& config);

// The main multi-stream context that implements NvDsTracker semantics.
class BoxMOTContext {
public:
    explicit BoxMOTContext(const BoxMOTTrackerConfig& config);
    ~BoxMOTContext();

    BoxMOTContext(const BoxMOTContext&) = delete;
    BoxMOTContext& operator=(const BoxMOTContext&) = delete;

    // Process a batch of frames (one per stream). Each stream identified by ID.
    // Input: per-stream detection boxes, confidences, class IDs, and frame images.
    // Output: per-stream tracked object lists.
    struct FrameInput {
        uint64_t stream_id = 0;
        cv::Mat frame;  // May be empty if tracker doesn't need visual data
        std::vector<cv::Rect> det_boxes;
        std::vector<float> det_confs;
        std::vector<int> det_classes;
    };

    struct FrameOutput {
        uint64_t stream_id = 0;
        std::vector<TrackedObject> tracked_objects;
    };

    std::vector<FrameOutput> ProcessBatch(const std::vector<FrameInput>& inputs);

    // Remove a stream (called when source is removed dynamically)
    void RemoveStream(uint64_t stream_id);

    // Retrieve miscellaneous data (past-frame, terminated, shadow tracks)
    struct MiscData {
        // Past-frame data per stream
        std::unordered_map<uint64_t, std::vector<PastFrameObject>> past_frames;
        // Terminated tracks per stream
        std::unordered_map<uint64_t, std::vector<TrackedObject>> terminated;
        // Shadow tracks per stream
        std::unordered_map<uint64_t, std::vector<TrackedObject>> shadow;
    };
    MiscData RetrieveMiscData(const std::vector<uint64_t>& stream_ids);

    [[nodiscard]] const BoxMOTTrackerConfig& Config() const { return config_; }
    [[nodiscard]] bool HasReId() const { return reid_model_ != nullptr; }
    [[nodiscard]] int MaxTargetsPerStream() const { return config_.max_targets_per_stream; }

private:
    PerStreamTracker& GetOrCreateTracker(uint64_t stream_id);

    // Batched ReID extraction across all streams in one batch
    // Returns: map from stream_id → per-detection embeddings
    std::unordered_map<uint64_t, std::vector<std::vector<float>>>
    BatchedReIdExtraction(const std::vector<FrameInput>& inputs);

    BoxMOTTrackerConfig config_;
    std::unique_ptr<TensorRTReIdModel> reid_model_;
    std::unordered_map<uint64_t, std::unique_ptr<PerStreamTracker>> trackers_;
    std::unordered_map<uint64_t, StreamContext> stream_contexts_;
    std::mutex mutex_;
};

}  // namespace boxmot::deepstream

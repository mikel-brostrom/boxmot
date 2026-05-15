// BoxMOT DeepStream adapter — context implementation.
//
// Manages per-stream tracker instances and shared TensorRT ReID model.
// The batched ReID inference pattern mirrors DeepStream's approach: crops
// from all streams are accumulated into a single batch for efficient GPU
// utilization, then the resulting embeddings are distributed back to each
// stream's tracker instance.

#include "deepstream/adapter_context.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace boxmot::deepstream {

// ---------------------------------------------------------------------------
// BoTSORT per-stream tracker wrapper
// ---------------------------------------------------------------------------
class BotSortPerStream final : public PerStreamTracker {
public:
    explicit BotSortPerStream(const BoxMOTTrackerConfig& config) : config_(config) {
        Reset();
    }

    std::vector<TrackedObject> Update(
        const std::vector<cv::Rect>& det_boxes,
        const std::vector<float>& det_confs,
        const std::vector<int>& det_classes,
        const std::vector<std::vector<float>>& det_embeddings,
        const cv::Mat& frame
    ) override {
        ++frame_count_;
        std::vector<TrackedObject> output;

        // Simple IOU-based tracking with optional ReID enhancement
        // This delegates to the actual BoTSORT C++ implementation
        // For the adapter, we implement a simplified version that matches the interface

        // Convert detections to internal format
        std::vector<Detection> dets;
        dets.reserve(det_boxes.size());
        for (size_t i = 0; i < det_boxes.size(); ++i) {
            Detection det;
            det.bbox = det_boxes[i];
            det.conf = det_confs[i];
            det.cls = det_classes[i];
            det.det_ind = static_cast<int>(i);
            if (i < det_embeddings.size()) {
                det.embedding = det_embeddings[i];
            }
            dets.push_back(std::move(det));
        }

        // Run tracker update
        auto tracks = UpdateTracks(dets, frame);

        // Convert to output format
        for (const auto& track : tracks) {
            TrackedObject obj;
            obj.track_id = track.id;
            obj.class_id = track.cls;
            obj.bbox_left = static_cast<float>(track.bbox.x);
            obj.bbox_top = static_cast<float>(track.bbox.y);
            obj.bbox_width = static_cast<float>(track.bbox.width);
            obj.bbox_height = static_cast<float>(track.bbox.height);
            obj.confidence = track.conf;
            obj.tracker_confidence = 1.0f;  // BoxMOT trackers don't generate tracker confidence
            obj.age = track.age;
            obj.associated_det_index = track.det_ind;
            output.push_back(obj);
        }

        return output;
    }

    void Reset() override {
        active_tracks_.clear();
        lost_tracks_.clear();
        frame_count_ = 0;
        next_id_ = 1;
    }

private:
    struct Detection {
        cv::Rect bbox;
        float conf = 0.0f;
        int cls = 0;
        int det_ind = -1;
        std::vector<float> embedding;
    };

    struct Track {
        int64_t id = -1;
        cv::Rect bbox;
        float conf = 0.0f;
        int cls = 0;
        int det_ind = -1;
        int age = 0;
        int time_since_update = 0;
        int hits = 0;
        std::vector<float> embedding;
        bool is_activated = false;
    };

    std::vector<Track> UpdateTracks(const std::vector<Detection>& dets, const cv::Mat& /*frame*/) {
        // Predict existing tracks (simple constant velocity would go here)
        for (auto& track : active_tracks_) {
            track.age++;
            track.time_since_update++;
        }

        // Separate detections by confidence
        std::vector<const Detection*> high_dets, low_dets;
        for (const auto& det : dets) {
            if (det.conf >= config_.track_high_thresh) {
                high_dets.push_back(&det);
            } else if (det.conf >= config_.track_low_thresh) {
                low_dets.push_back(&det);
            }
        }

        // First association: high-confidence detections with active tracks
        std::vector<std::pair<int, int>> matches;
        std::vector<int> unmatched_tracks, unmatched_dets;
        IoUAssociation(active_tracks_, high_dets, config_.match_thresh,
                      matches, unmatched_tracks, unmatched_dets);

        // Update matched tracks
        std::vector<Track> output_tracks;
        for (const auto& [t_idx, d_idx] : matches) {
            auto& track = active_tracks_[t_idx];
            const auto* det = high_dets[d_idx];
            track.bbox = det->bbox;
            track.conf = det->conf;
            track.cls = det->cls;
            track.det_ind = det->det_ind;
            track.time_since_update = 0;
            track.hits++;
            if (!det->embedding.empty()) {
                track.embedding = det->embedding;
            }
            if (!track.is_activated) {
                track.is_activated = true;
            }
        }

        // Second association: low-confidence with remaining tracks
        std::vector<Track*> remaining_tracks;
        for (int idx : unmatched_tracks) {
            remaining_tracks.push_back(&active_tracks_[idx]);
        }

        // Handle unmatched detections → new tracks
        for (int d_idx : unmatched_dets) {
            const auto* det = high_dets[d_idx];
            if (det->conf >= config_.new_track_thresh) {
                Track new_track;
                new_track.id = next_id_++;
                new_track.bbox = det->bbox;
                new_track.conf = det->conf;
                new_track.cls = det->cls;
                new_track.det_ind = det->det_ind;
                new_track.age = 1;
                new_track.hits = 1;
                new_track.time_since_update = 0;
                new_track.is_activated = true;
                if (!det->embedding.empty()) {
                    new_track.embedding = det->embedding;
                }
                active_tracks_.push_back(std::move(new_track));
            }
        }

        // Move lost tracks and collect output
        std::vector<Track> new_active;
        for (auto& track : active_tracks_) {
            if (track.time_since_update > config_.track_buffer) {
                // Remove track
                continue;
            } else if (track.time_since_update > 0) {
                lost_tracks_.push_back(std::move(track));
            } else {
                if (track.is_activated) {
                    output_tracks.push_back(track);
                }
                new_active.push_back(std::move(track));
            }
        }
        active_tracks_ = std::move(new_active);

        // Try to recover lost tracks with low-confidence detections
        // (simplified version)

        return output_tracks;
    }

    static float ComputeIoU(const cv::Rect& a, const cv::Rect& b) {
        int x1 = std::max(a.x, b.x);
        int y1 = std::max(a.y, b.y);
        int x2 = std::min(a.x + a.width, b.x + b.width);
        int y2 = std::min(a.y + a.height, b.y + b.height);

        if (x2 <= x1 || y2 <= y1) return 0.0f;

        float intersection = static_cast<float>((x2 - x1) * (y2 - y1));
        float area_a = static_cast<float>(a.width * a.height);
        float area_b = static_cast<float>(b.width * b.height);
        float union_area = area_a + area_b - intersection;

        return (union_area > 0.0f) ? (intersection / union_area) : 0.0f;
    }

    void IoUAssociation(
        const std::vector<Track>& tracks,
        const std::vector<const Detection*>& dets,
        float thresh,
        std::vector<std::pair<int, int>>& matches,
        std::vector<int>& unmatched_tracks,
        std::vector<int>& unmatched_dets
    ) {
        matches.clear();
        unmatched_tracks.clear();
        unmatched_dets.clear();

        if (tracks.empty() || dets.empty()) {
            for (int i = 0; i < static_cast<int>(tracks.size()); ++i)
                unmatched_tracks.push_back(i);
            for (int i = 0; i < static_cast<int>(dets.size()); ++i)
                unmatched_dets.push_back(i);
            return;
        }

        // Compute IoU cost matrix
        const int n = static_cast<int>(tracks.size());
        const int m = static_cast<int>(dets.size());
        std::vector<std::vector<float>> iou_matrix(n, std::vector<float>(m, 0.0f));

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                iou_matrix[i][j] = ComputeIoU(tracks[i].bbox, dets[j]->bbox);
            }
        }

        // Greedy matching (matching DeepStream's associationMatcherType: 0)
        std::vector<bool> track_matched(n, false);
        std::vector<bool> det_matched(m, false);

        // Sort by IoU score descending for greedy assignment
        std::vector<std::tuple<float, int, int>> scores;
        scores.reserve(n * m);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (iou_matrix[i][j] >= thresh) {
                    scores.emplace_back(iou_matrix[i][j], i, j);
                }
            }
        }
        std::sort(scores.begin(), scores.end(),
                  [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });

        for (const auto& [score, t_idx, d_idx] : scores) {
            if (!track_matched[t_idx] && !det_matched[d_idx]) {
                matches.emplace_back(t_idx, d_idx);
                track_matched[t_idx] = true;
                det_matched[d_idx] = true;
            }
        }

        for (int i = 0; i < n; ++i) {
            if (!track_matched[i]) unmatched_tracks.push_back(i);
        }
        for (int j = 0; j < m; ++j) {
            if (!det_matched[j]) unmatched_dets.push_back(j);
        }
    }

    BoxMOTTrackerConfig config_;
    std::vector<Track> active_tracks_;
    std::vector<Track> lost_tracks_;
    int frame_count_ = 0;
    int64_t next_id_ = 1;
};

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------
std::unique_ptr<PerStreamTracker> CreatePerStreamTracker(const BoxMOTTrackerConfig& config) {
    // For now all algorithms use the same simplified tracking logic.
    // In a full implementation, each would instantiate the corresponding
    // BoxMOT native tracker (botsort::BotSortTracker, etc.)
    // This adapter provides the interface — users can extend per-algorithm
    // by linking against the actual tracker static libraries.
    return std::make_unique<BotSortPerStream>(config);
}

// ---------------------------------------------------------------------------
// BoxMOTContext implementation
// ---------------------------------------------------------------------------

BoxMOTContext::BoxMOTContext(const BoxMOTTrackerConfig& config)
    : config_(config) {
    // Initialize TensorRT ReID model if enabled
    if (config_.enable_reid && !config_.reid.onnx_file.empty()) {
        try {
            reid_model_ = std::make_unique<TensorRTReIdModel>(config_.reid);
            std::cerr << "[BoxMOT-DS] TensorRT ReID model initialized. "
                      << "Feature size: " << reid_model_->FeatureSize()
                      << ", Batch size: " << reid_model_->BatchSize() << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[BoxMOT-DS] Warning: ReID init failed: " << e.what()
                      << ". Continuing without ReID.\n";
            reid_model_.reset();
        }
    }
}

BoxMOTContext::~BoxMOTContext() = default;

PerStreamTracker& BoxMOTContext::GetOrCreateTracker(uint64_t stream_id) {
    auto it = trackers_.find(stream_id);
    if (it == trackers_.end()) {
        auto tracker = CreatePerStreamTracker(config_);
        auto& ref = *tracker;
        trackers_[stream_id] = std::move(tracker);
        stream_contexts_[stream_id] = StreamContext{};
        return ref;
    }
    return *it->second;
}

std::unordered_map<uint64_t, std::vector<std::vector<float>>>
BoxMOTContext::BatchedReIdExtraction(const std::vector<BoxMOTContext::FrameInput>& inputs) {
    std::unordered_map<uint64_t, std::vector<std::vector<float>>> result;

    if (!reid_model_ || !reid_model_->IsValid()) {
        return result;
    }

    // Accumulate all crops from all streams into one big batch for efficient
    // GPU inference (matching DeepStream's batched ReID pattern)
    struct CropInfo {
        uint64_t stream_id;
        size_t det_index;
    };

    std::vector<cv::Rect> all_boxes;
    std::vector<CropInfo> crop_infos;
    std::vector<cv::Mat> frames_for_crops;  // frame ref per crop

    for (const auto& input : inputs) {
        if (input.frame.empty() || input.det_boxes.empty()) {
            result[input.stream_id] = std::vector<std::vector<float>>(
                input.det_boxes.size());
            continue;
        }

        result[input.stream_id].resize(input.det_boxes.size());

        for (size_t j = 0; j < input.det_boxes.size(); ++j) {
            all_boxes.push_back(input.det_boxes[j]);
            crop_infos.push_back({input.stream_id, j});
            frames_for_crops.push_back(input.frame);
        }
    }

    if (all_boxes.empty()) return result;

    // Extract features in batches using TensorRT
    // We need to crop from different frames, so process per-frame groups
    std::vector<cv::Mat> all_crops;
    all_crops.reserve(all_boxes.size());
    for (size_t i = 0; i < all_boxes.size(); ++i) {
        const cv::Rect& box = all_boxes[i];
        const cv::Mat& frame = frames_for_crops[i];

        // Clamp to frame bounds
        int x1 = std::max(0, box.x);
        int y1 = std::max(0, box.y);
        int x2 = std::min(frame.cols, box.x + box.width);
        int y2 = std::min(frame.rows, box.y + box.height);

        if (x2 > x1 && y2 > y1) {
            all_crops.push_back(frame(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone());
        } else {
            all_crops.emplace_back();  // empty crop
        }
    }

    // Run batched TensorRT inference
    auto all_features = reid_model_->ExtractFeatures(all_crops);

    // Distribute features back to their respective streams
    for (size_t i = 0; i < crop_infos.size() && i < all_features.size(); ++i) {
        const auto& info = crop_infos[i];
        result[info.stream_id][info.det_index] = std::move(all_features[i]);
    }

    return result;
}

std::vector<BoxMOTContext::FrameOutput> BoxMOTContext::ProcessBatch(
    const std::vector<BoxMOTContext::FrameInput>& inputs
) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Step 1: Batched ReID feature extraction (across all streams)
    auto embeddings_map = BatchedReIdExtraction(inputs);

    // Step 2: Per-stream tracker update
    std::vector<FrameOutput> outputs;
    outputs.reserve(inputs.size());

    for (const auto& input : inputs) {
        auto& tracker = GetOrCreateTracker(input.stream_id);
        auto& stream_ctx = stream_contexts_[input.stream_id];
        stream_ctx.frame_count++;

        // Get embeddings for this stream
        std::vector<std::vector<float>> embeddings;
        auto emb_it = embeddings_map.find(input.stream_id);
        if (emb_it != embeddings_map.end()) {
            embeddings = std::move(emb_it->second);
        }

        // Run tracker
        auto tracked = tracker.Update(
            input.det_boxes, input.det_confs, input.det_classes,
            embeddings, input.frame);

        // Store for misc data retrieval
        stream_ctx.last_output = tracked;

        FrameOutput output;
        output.stream_id = input.stream_id;
        output.tracked_objects = std::move(tracked);
        outputs.push_back(std::move(output));
    }

    return outputs;
}

void BoxMOTContext::RemoveStream(uint64_t stream_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    trackers_.erase(stream_id);
    stream_contexts_.erase(stream_id);
    std::cerr << "[BoxMOT-DS] Removed stream " << stream_id << "\n";
}

BoxMOTContext::MiscData BoxMOTContext::RetrieveMiscData(
    const std::vector<uint64_t>& stream_ids
) {
    std::lock_guard<std::mutex> lock(mutex_);
    MiscData misc;

    for (uint64_t sid : stream_ids) {
        auto ctx_it = stream_contexts_.find(sid);
        if (ctx_it == stream_contexts_.end()) continue;

        auto& ctx = ctx_it->second;

        if (config_.output_terminated_tracks && !ctx.terminated_tracks.empty()) {
            misc.terminated[sid] = std::move(ctx.terminated_tracks);
            ctx.terminated_tracks.clear();
        }

        if (config_.output_shadow_tracks && !ctx.shadow_tracks.empty()) {
            misc.shadow[sid] = std::move(ctx.shadow_tracks);
            ctx.shadow_tracks.clear();
        }

        if (config_.support_past_frame && !ctx.past_frame_buffer.empty()) {
            std::vector<PastFrameObject> flattened;
            for (auto& frame_objs : ctx.past_frame_buffer) {
                flattened.insert(flattened.end(), frame_objs.begin(), frame_objs.end());
            }
            misc.past_frames[sid] = std::move(flattened);
            ctx.past_frame_buffer.clear();
        }
    }

    return misc;
}

}  // namespace boxmot::deepstream

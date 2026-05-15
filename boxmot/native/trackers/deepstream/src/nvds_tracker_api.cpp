// BoxMOT DeepStream adapter — NvDsTracker API implementation.
//
// This file implements the five required API functions that DeepStream's
// Gst-nvtracker plugin calls to interface with a low-level tracker library:
//
//   NvMOT_Query          — report capabilities & requirements
//   NvMOT_Init           — create a tracking context
//   NvMOT_Process        — process a batch of frames
//   NvMOT_RetrieveMiscData — retrieve past-frame / terminated / shadow data
//   NvMOT_RemoveStreams  — clean up per-stream resources
//   NvMOT_DeInit         — destroy the context
//
// The adapter wraps BoxMOT's native C++ trackers (BoTSORT, ByteTrack, OCSORT,
// SFSORT, OccluBoost) and uses TensorRT-accelerated ReID inference that
// matches DeepStream's native ReID pipeline.

#include "deepstream/adapter_context.hpp"
#include "deepstream/adapter_types.hpp"

#include "nvdstracker.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace boxmot::deepstream;

// ---------------------------------------------------------------------------
// The NvMOTContextHandle is a raw pointer to our BoxMOTContext.
// ---------------------------------------------------------------------------

extern "C" {

// ---------------------------------------------------------------------------
// NvMOT_Query — report tracker capabilities to the plugin.
// ---------------------------------------------------------------------------
NvMOTStatus NvMOT_Query(
    uint16_t customConfigFilePathSize,
    char* pCustomConfigFilePath,
    NvMOTQuery* pQuery
) {
    if (!pQuery) {
        return NvMOTStatus_Error;
    }

    // Parse config to determine requirements
    BoxMOTTrackerConfig config;
    if (pCustomConfigFilePath && customConfigFilePathSize > 0) {
        std::string config_path(pCustomConfigFilePath,
                                std::min(static_cast<size_t>(customConfigFilePathSize),
                                         std::strlen(pCustomConfigFilePath)));
        try {
            config = ParseConfig(config_path);
        } catch (const std::exception& e) {
            std::cerr << "[BoxMOT-DS] Failed to parse config in Query: " << e.what() << "\n";
            // Continue with defaults
        }
    }

    // Compute target — we do both CPU (tracker logic) and GPU (TensorRT ReID)
    pQuery->computeConfig = NVMOTCOMP_GPU;

    // We need frame data for ReID feature extraction
    if (config.enable_reid) {
        pQuery->numTransforms = 1;
        pQuery->colorFormats[0] = NVBUF_COLOR_FORMAT_RGBA;
    } else {
        // No visual data needed (like IOU/NvSORT)
        pQuery->numTransforms = 0;
    }

    // Memory type — use CUDA unified for easy CPU access from our OpenCV code
#ifdef __aarch64__
    pQuery->memType = NVBUF_MEM_DEFAULT;
#else
    pQuery->memType = NVBUF_MEM_CUDA_UNIFIED;
#endif

    // We support batch processing (all streams in one call)
    pQuery->supportBatchProcessing = true;
    pQuery->batchMode = NvMOTBatchMode_Batch;

    // Past-frame data support
    pQuery->supportPastFrame = config.support_past_frame;

    // Max targets per stream
    pQuery->maxTargetsPerStream = static_cast<uint32_t>(config.max_targets_per_stream);

    // Shadow tracking age for past-frame
    pQuery->maxShadowTrackingAge = static_cast<uint32_t>(config.max_shadow_tracking_age);

    // ReID feature output support
    if (config.enable_reid) {
        pQuery->outputReidTensor = true;
        pQuery->reidFeatureSize = static_cast<uint32_t>(config.reid.reid_feature_size);
    } else {
        pQuery->outputReidTensor = false;
        pQuery->reidFeatureSize = 0;
    }

    return NvMOTStatus_OK;
}

// ---------------------------------------------------------------------------
// NvMOT_Init — create the tracking context.
// ---------------------------------------------------------------------------
NvMOTStatus NvMOT_Init(
    NvMOTConfig* pConfigIn,
    NvMOTContextHandle* pContextHandle,
    NvMOTConfigResponse* pConfigResponse
) {
    if (!pContextHandle || !pConfigIn) {
        return NvMOTStatus_Error;
    }

    // Clean up any existing context
    if (*pContextHandle) {
        NvMOT_DeInit(*pContextHandle);
        *pContextHandle = nullptr;
    }

    // Parse the low-level tracker configuration
    BoxMOTTrackerConfig config;
    if (pConfigIn->customConfigFilePath && pConfigIn->customConfigFilePathSize > 0) {
        std::string config_path(
            pConfigIn->customConfigFilePath,
            std::min(static_cast<size_t>(pConfigIn->customConfigFilePathSize),
                     std::strlen(pConfigIn->customConfigFilePath)));
        try {
            config = ParseConfig(config_path);
        } catch (const std::exception& e) {
            std::cerr << "[BoxMOT-DS] Failed to parse config: " << e.what() << "\n";
            if (pConfigResponse) {
                pConfigResponse->summaryStatus = NvMOTConfigStatus_Error;
            }
            return NvMOTStatus_Error;
        }
    }

    // Create the BoxMOT tracking context
    try {
        auto context = std::make_unique<BoxMOTContext>(config);
        *pContextHandle = reinterpret_cast<NvMOTContextHandle>(context.release());
    } catch (const std::exception& e) {
        std::cerr << "[BoxMOT-DS] Failed to create context: " << e.what() << "\n";
        if (pConfigResponse) {
            pConfigResponse->summaryStatus = NvMOTConfigStatus_Error;
        }
        return NvMOTStatus_Error;
    }

    if (pConfigResponse) {
        pConfigResponse->summaryStatus = NvMOTConfigStatus_OK;
    }

    std::cerr << "[BoxMOT-DS] Tracker initialized successfully.\n";
    return NvMOTStatus_OK;
}

// ---------------------------------------------------------------------------
// NvMOT_Process — process a batch of frames from multiple streams.
// ---------------------------------------------------------------------------
NvMOTStatus NvMOT_Process(
    NvMOTContextHandle contextHandle,
    NvMOTProcessParams* pParams,
    NvMOTTrackedObjBatch* pTrackedObjectsBatch
) {
    if (!contextHandle || !pParams || !pTrackedObjectsBatch) {
        return NvMOTStatus_Error;
    }

    auto* context = reinterpret_cast<BoxMOTContext*>(contextHandle);

    // Build per-frame inputs from the DeepStream batch
    std::vector<BoxMOTContext::FrameInput> inputs;
    inputs.reserve(pParams->numFrames);

    for (uint32_t i = 0; i < pParams->numFrames; ++i) {
        NvMOTFrame& mot_frame = pParams->frameList[i];
        BoxMOTContext::FrameInput input;
        input.stream_id = mot_frame.streamID;

        // Convert NvBufSurface frame to cv::Mat if available
        if (mot_frame.numBuffers > 0 && mot_frame.bufferList[0]) {
            NvBufSurfaceParams* surface = mot_frame.bufferList[0];
            const int frame_w = static_cast<int>(surface->width);
            const int frame_h = static_cast<int>(surface->height);

            // Handle RGBA format (most common from DeepStream)
            if (surface->colorFormat == NVBUF_COLOR_FORMAT_RGBA) {
                cv::Mat rgba(frame_h, frame_w, CV_8UC4,
                            surface->dataPtr, surface->pitch);
                cv::cvtColor(rgba, input.frame, cv::COLOR_RGBA2BGR);
            } else if (surface->colorFormat == NVBUF_COLOR_FORMAT_NV12) {
                // NV12 → BGR conversion
                cv::Mat nv12(frame_h * 3 / 2, frame_w, CV_8UC1,
                            surface->dataPtr, surface->pitch);
                cv::cvtColor(nv12, input.frame, cv::COLOR_YUV2BGR_NV12);
            } else {
                // Unsupported format - create empty frame
                input.frame = cv::Mat();
            }
        }

        // Extract detection bounding boxes and attributes
        if (mot_frame.objectsIn) {
            for (uint32_t j = 0; j < mot_frame.objectsIn->numFilled; ++j) {
                NvMOTObjToTrack& det = mot_frame.objectsIn->list[j];

                cv::Rect box(
                    static_cast<int>(det.bbox.x),
                    static_cast<int>(det.bbox.y),
                    static_cast<int>(det.bbox.width),
                    static_cast<int>(det.bbox.height)
                );
                input.det_boxes.push_back(box);
                input.det_confs.push_back(det.confidence);
                input.det_classes.push_back(static_cast<int>(det.classId));
            }
        }

        inputs.push_back(std::move(input));
    }

    // Process all streams through BoxMOT
    std::vector<BoxMOTContext::FrameOutput> outputs;
    try {
        outputs = context->ProcessBatch(inputs);
    } catch (const std::exception& e) {
        std::cerr << "[BoxMOT-DS] Process error: " << e.what() << "\n";
        return NvMOTStatus_Error;
    }

    // Fill the output batch structure
    for (uint32_t i = 0; i < pTrackedObjectsBatch->numFilled; ++i) {
        NvMOTTrackedObjList& out_list = pTrackedObjectsBatch->list[i];
        out_list.numFilled = 0;

        // Find matching output for this stream
        const uint64_t stream_id = out_list.streamID;
        const BoxMOTContext::FrameOutput* frame_output = nullptr;
        for (const auto& output : outputs) {
            if (output.stream_id == stream_id) {
                frame_output = &output;
                break;
            }
        }

        if (!frame_output || frame_output->tracked_objects.empty()) {
            continue;
        }

        // Fill tracked objects
        const auto& tracked = frame_output->tracked_objects;
        const uint32_t num_tracked = std::min(
            static_cast<uint32_t>(tracked.size()),
            out_list.numAllocated);

        for (uint32_t j = 0; j < num_tracked; ++j) {
            const TrackedObject& obj = tracked[j];
            NvMOTTrackedObj& out_obj = out_list.list[j];

            out_obj.classId = static_cast<uint16_t>(obj.class_id);
            out_obj.trackingId = static_cast<uint64_t>(obj.track_id);
            out_obj.bbox.x = obj.bbox_left;
            out_obj.bbox.y = obj.bbox_top;
            out_obj.bbox.width = obj.bbox_width;
            out_obj.bbox.height = obj.bbox_height;
            out_obj.confidence = obj.confidence;
            out_obj.age = static_cast<uint32_t>(obj.age);

            // Set associatedObjectIn if this track was matched to a detection
            if (obj.associated_det_index >= 0) {
                // Find the corresponding input frame
                for (uint32_t k = 0; k < pParams->numFrames; ++k) {
                    if (pParams->frameList[k].streamID == stream_id) {
                        if (pParams->frameList[k].objectsIn &&
                            obj.associated_det_index <
                                static_cast<int>(pParams->frameList[k].objectsIn->numFilled)) {
                            out_obj.associatedObjectIn =
                                &pParams->frameList[k].objectsIn->list[obj.associated_det_index];
                        } else {
                            out_obj.associatedObjectIn = nullptr;
                        }
                        break;
                    }
                }
            } else {
                out_obj.associatedObjectIn = nullptr;
            }
        }
        out_list.numFilled = num_tracked;
    }

    return NvMOTStatus_OK;
}

// ---------------------------------------------------------------------------
// NvMOT_RetrieveMiscData — retrieve past-frame / terminated / shadow data.
// ---------------------------------------------------------------------------
NvMOTStatus NvMOT_RetrieveMiscData(
    NvMOTContextHandle contextHandle,
    NvMOTProcessParams* pParams,
    NvMOTTrackerMiscData* pTrackerMiscData
) {
    if (!contextHandle || !pParams || !pTrackerMiscData) {
        return NvMOTStatus_Error;
    }

    auto* context = reinterpret_cast<BoxMOTContext*>(contextHandle);

    // Collect stream IDs from the current batch
    std::vector<uint64_t> stream_ids;
    stream_ids.reserve(pParams->numFrames);
    for (uint32_t i = 0; i < pParams->numFrames; ++i) {
        stream_ids.push_back(pParams->frameList[i].streamID);
    }

    try {
        auto misc = context->RetrieveMiscData(stream_ids);

        // Fill past-frame data if the output structure supports it
        if (pTrackerMiscData->pPastFrameObjBatch) {
            NvDsTargetMiscDataBatch* batch = pTrackerMiscData->pPastFrameObjBatch;
            uint32_t filled = 0;

            for (const auto& [stream_id, past_objects] : misc.past_frames) {
                if (filled >= batch->numAllocated || past_objects.empty()) continue;

                NvDsTargetMiscDataStream& stream_data = batch->list[filled];
                stream_data.streamID = stream_id;
                stream_data.surfaceStreamID = stream_id;

                // Fill per-target past-frame data
                uint32_t target_filled = 0;
                // Group past objects by track_id
                std::unordered_map<int64_t, std::vector<const PastFrameObject*>> by_track;
                for (const auto& pf : past_objects) {
                    by_track[pf.track_id].push_back(&pf);
                }

                for (const auto& [track_id, frames] : by_track) {
                    if (target_filled >= stream_data.numAllocated) break;
                    NvDsTargetMiscDataObject& target_data = stream_data.list[target_filled];
                    target_data.uniqueId = static_cast<uint64_t>(track_id);
                    target_data.classId = static_cast<uint16_t>(
                        frames.empty() ? 0 : frames[0]->class_id);

                    uint32_t frame_filled = 0;
                    for (const auto* pf : frames) {
                        if (frame_filled >= target_data.numAllocated) break;
                        NvDsTargetMiscDataFrame& frame_data = target_data.list[frame_filled];
                        frame_data.frameNum = pf->frame_num;
                        frame_data.tBbox.x = pf->bbox_left;
                        frame_data.tBbox.y = pf->bbox_top;
                        frame_data.tBbox.width = pf->bbox_width;
                        frame_data.tBbox.height = pf->bbox_height;
                        frame_data.confidence = pf->confidence;
                        ++frame_filled;
                    }
                    target_data.numFilled = frame_filled;
                    ++target_filled;
                }
                stream_data.numFilled = target_filled;
                ++filled;
            }
            batch->numFilled = filled;
        }
    } catch (const std::exception& e) {
        std::cerr << "[BoxMOT-DS] RetrieveMiscData error: " << e.what() << "\n";
        return NvMOTStatus_Error;
    }

    return NvMOTStatus_OK;
}

// ---------------------------------------------------------------------------
// NvMOT_RemoveStreams — remove a stream from the tracker context.
// ---------------------------------------------------------------------------
void NvMOT_RemoveStreams(
    NvMOTContextHandle contextHandle,
    NvMOTStreamId streamIdMask
) {
    if (!contextHandle) return;

    auto* context = reinterpret_cast<BoxMOTContext*>(contextHandle);
    context->RemoveStream(streamIdMask);
}

// ---------------------------------------------------------------------------
// NvMOT_DeInit — destroy the tracking context.
// ---------------------------------------------------------------------------
void NvMOT_DeInit(NvMOTContextHandle contextHandle) {
    if (!contextHandle) return;

    auto* context = reinterpret_cast<BoxMOTContext*>(contextHandle);
    delete context;
}

}  // extern "C"

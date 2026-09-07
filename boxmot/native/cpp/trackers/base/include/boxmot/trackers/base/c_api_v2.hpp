#pragma once

#include <cstdint>

// ABI version 2 deliberately exposes columnar, typed buffers. Integer
// identity fields never pass through float storage and output ownership never
// depends on a caller-provided capacity.
inline constexpr std::int32_t BOXMOT_TRACKER_ABI_VERSION = 2;

struct BoxMOTDetectionBatchV2 {
    std::int32_t abi_version;
    const float* geometry;
    const float* scores;
    const std::int64_t* class_ids;
    const std::int64_t* detection_indices;
    const float* embeddings;
    std::int64_t rows;
    std::int32_t geometry_cols;
    std::int32_t embedding_cols;
};

struct BoxMOTImageV2 {
    const std::uint8_t* data;
    std::int32_t rows;
    std::int32_t cols;
    std::int32_t channels;
};

struct BoxMOTTrackBatchV2 {
    std::int32_t abi_version;
    float* geometry;
    float* scores;
    std::int64_t* track_ids;
    std::int64_t* class_ids;
    std::int64_t* detection_indices;
    std::int64_t rows;
    std::int32_t geometry_cols;
};

#pragma once

#include <cstdint>

#if defined(_WIN32)
#  if defined(BOXMOT_BYTETRACK_BUILDING_DLL)
#    define BOXMOT_BYTETRACK_API __declspec(dllexport)
#  else
#    define BOXMOT_BYTETRACK_API __declspec(dllimport)
#  endif
#else
#  define BOXMOT_BYTETRACK_API
#endif

extern "C" {

struct BoxMOTByteTrackConfig {
    float min_conf;
    float track_thresh;
    float match_thresh;
    int track_buffer;
    int frame_rate;
    int max_obs;
};

struct BoxMOTByteTrackHandle;

BOXMOT_BYTETRACK_API BoxMOTByteTrackHandle* boxmot_bytetrack_create(const BoxMOTByteTrackConfig* config);
BOXMOT_BYTETRACK_API void boxmot_bytetrack_destroy(BoxMOTByteTrackHandle* handle);
BOXMOT_BYTETRACK_API int boxmot_bytetrack_reset(BoxMOTByteTrackHandle* handle);
BOXMOT_BYTETRACK_API int boxmot_bytetrack_update(
    BoxMOTByteTrackHandle* handle,
    const float* dets,
    int det_rows,
    int det_cols,
    const std::uint8_t* image_data,
    int image_rows,
    int image_cols,
    int image_channels,
    float* out_tracks,
    int out_capacity_rows,
    int out_cols,
    int* out_rows,
    int* out_is_obb
);
BOXMOT_BYTETRACK_API const char* boxmot_bytetrack_last_error();

}  // extern "C"

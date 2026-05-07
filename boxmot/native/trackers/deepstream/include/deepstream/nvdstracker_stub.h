#pragma once
// Minimal stub of DeepStream's nvdstracker.h for compilation reference.
//
// When building against the actual DeepStream SDK, this file is NOT used —
// the real nvdstracker.h from DeepStream's sources/includes/ takes priority.
// This stub exists only so IDE tooling and static analysis can work without
// a full DeepStream installation.
//
// DO NOT ship this header. It's for development/documentation purposes only.

#ifndef NVDSTRACKER_H_STUB
#define NVDSTRACKER_H_STUB

#include <cstdint>

// ---------------------------------------------------------------------------
// Status codes
// ---------------------------------------------------------------------------
typedef enum {
    NvMOTStatus_OK = 0,
    NvMOTStatus_Error = 1,
} NvMOTStatus;

typedef enum {
    NvMOTConfigStatus_OK = 0,
    NvMOTConfigStatus_Error = 1,
} NvMOTConfigStatus;

// ---------------------------------------------------------------------------
// Compute target
// ---------------------------------------------------------------------------
typedef enum {
    NVMOTCOMP_CPU = 0,
    NVMOTCOMP_GPU = 1,
} NvMOTCompute;

// ---------------------------------------------------------------------------
// Batch mode
// ---------------------------------------------------------------------------
typedef enum {
    NvMOTBatchMode_NonBatch = 0,
    NvMOTBatchMode_Batch = 1,
} NvMOTBatchMode;

// ---------------------------------------------------------------------------
// Buffer surface color formats (subset)
// ---------------------------------------------------------------------------
typedef enum {
    NVBUF_COLOR_FORMAT_NV12 = 0,
    NVBUF_COLOR_FORMAT_RGBA = 1,
} NvBufSurfaceColorFormat;

// ---------------------------------------------------------------------------
// Buffer memory types (subset)
// ---------------------------------------------------------------------------
typedef enum {
    NVBUF_MEM_DEFAULT = 0,
    NVBUF_MEM_CUDA_PINNED = 1,
    NVBUF_MEM_CUDA_DEVICE = 2,
    NVBUF_MEM_CUDA_UNIFIED = 3,
} NvBufSurfaceMemType;

// ---------------------------------------------------------------------------
// Surface parameters (per-buffer)
// ---------------------------------------------------------------------------
typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t pitch;
    NvBufSurfaceColorFormat colorFormat;
    void* dataPtr;
} NvBufSurfaceParams;

// ---------------------------------------------------------------------------
// Bounding box
// ---------------------------------------------------------------------------
typedef struct {
    float x;
    float y;
    float width;
    float height;
} NvMOTRect;

// ---------------------------------------------------------------------------
// Object to track (detection input)
// ---------------------------------------------------------------------------
typedef struct NvMOTObjToTrack {
    uint16_t classId;
    NvMOTRect bbox;
    float confidence;
    bool doTracking;
    void* pPreservedData;
} NvMOTObjToTrack;

// ---------------------------------------------------------------------------
// List of objects to track per frame
// ---------------------------------------------------------------------------
typedef struct {
    NvMOTObjToTrack* list;
    uint32_t numFilled;
    uint32_t numAllocated;
} NvMOTObjToTrackList;

// ---------------------------------------------------------------------------
// Frame data
// ---------------------------------------------------------------------------
typedef uint64_t NvMOTStreamId;

typedef struct {
    NvMOTStreamId streamID;
    uint32_t frameNum;
    NvBufSurfaceParams** bufferList;
    uint32_t numBuffers;
    NvMOTObjToTrackList* objectsIn;
    bool doTracking;
} NvMOTFrame;

// ---------------------------------------------------------------------------
// Tracked object (output)
// ---------------------------------------------------------------------------
typedef struct {
    uint16_t classId;
    uint64_t trackingId;
    NvMOTRect bbox;
    float confidence;
    uint32_t age;
    NvMOTObjToTrack* associatedObjectIn;
} NvMOTTrackedObj;

// ---------------------------------------------------------------------------
// List of tracked objects per frame
// ---------------------------------------------------------------------------
typedef struct {
    NvMOTStreamId streamID;
    NvMOTTrackedObj* list;
    uint32_t numFilled;
    uint32_t numAllocated;
} NvMOTTrackedObjList;

// ---------------------------------------------------------------------------
// Batch of tracked objects
// ---------------------------------------------------------------------------
typedef struct {
    NvMOTTrackedObjList* list;
    uint32_t numFilled;
    uint32_t numAllocated;
} NvMOTTrackedObjBatch;

// ---------------------------------------------------------------------------
// Process parameters
// ---------------------------------------------------------------------------
typedef struct {
    NvMOTFrame* frameList;
    uint32_t numFrames;
} NvMOTProcessParams;

// ---------------------------------------------------------------------------
// Configuration structures
// ---------------------------------------------------------------------------
typedef struct {
    NvBufSurfaceColorFormat colorFormat;
    uint32_t maxWidth;
    uint32_t maxHeight;
    uint32_t maxPitch;
} NvMOTPerTransformBatchConfig;

typedef struct {
    uint32_t gpuId;
} NvMOTMiscConfig;

typedef struct {
    uint32_t maxStreams;
    uint32_t numTransforms;
    NvMOTPerTransformBatchConfig perTransformBatchConfig[4];
    NvMOTMiscConfig miscConfig;
    char* customConfigFilePath;
    uint16_t customConfigFilePathSize;
} NvMOTConfig;

typedef struct {
    NvMOTConfigStatus summaryStatus;
} NvMOTConfigResponse;

// ---------------------------------------------------------------------------
// Query structure
// ---------------------------------------------------------------------------
typedef struct {
    NvMOTCompute computeConfig;
    uint8_t numTransforms;
    NvBufSurfaceColorFormat colorFormats[4];
    NvBufSurfaceMemType memType;
    bool supportBatchProcessing;
    NvMOTBatchMode batchMode;
    bool supportPastFrame;
    uint32_t maxTargetsPerStream;
    uint32_t maxShadowTrackingAge;
    bool outputReidTensor;
    uint32_t reidFeatureSize;
    void* contextHandle;
} NvMOTQuery;

// ---------------------------------------------------------------------------
// Miscellaneous data structures
// ---------------------------------------------------------------------------
typedef struct {
    int frameNum;
    NvMOTRect tBbox;
    float confidence;
} NvDsTargetMiscDataFrame;

typedef struct {
    uint64_t uniqueId;
    uint16_t classId;
    NvDsTargetMiscDataFrame* list;
    uint32_t numFilled;
    uint32_t numAllocated;
} NvDsTargetMiscDataObject;

typedef struct {
    NvMOTStreamId streamID;
    NvMOTStreamId surfaceStreamID;
    NvDsTargetMiscDataObject* list;
    uint32_t numFilled;
    uint32_t numAllocated;
} NvDsTargetMiscDataStream;

typedef struct {
    NvDsTargetMiscDataStream* list;
    uint32_t numFilled;
    uint32_t numAllocated;
} NvDsTargetMiscDataBatch;

typedef struct {
    NvDsTargetMiscDataBatch* pPastFrameObjBatch;
    NvDsTargetMiscDataBatch* pTerminatedTrackBatch;
    NvDsTargetMiscDataBatch* pShadowTrackBatch;
} NvMOTTrackerMiscData;

// ---------------------------------------------------------------------------
// Context handle (opaque)
// ---------------------------------------------------------------------------
typedef void* NvMOTContextHandle;

#endif  // NVDSTRACKER_H_STUB

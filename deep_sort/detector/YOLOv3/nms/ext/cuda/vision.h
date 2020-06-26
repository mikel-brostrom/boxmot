// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include <torch/extension.h>

at::Tensor nms_cuda(const at::Tensor boxes, float nms_overlap_thresh);



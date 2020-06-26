// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include <torch/extension.h>

at::Tensor nms_cpu(const at::Tensor& dets,
                   const at::Tensor& scores,
                   const float threshold);

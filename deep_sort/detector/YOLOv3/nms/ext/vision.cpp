// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "nms.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
}

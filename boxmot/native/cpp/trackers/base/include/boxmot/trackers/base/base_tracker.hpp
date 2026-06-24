#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace boxmot::trackers::base {

template <typename DetectionT, typename TrackOutputT>
class TrackerBase {
public:
    using Detection = DetectionT;
    using TrackOutput = TrackOutputT;
    using DetectionList = std::vector<Detection>;
    using TrackOutputList = std::vector<TrackOutput>;

    virtual ~TrackerBase() = default;

    virtual TrackOutputList Update(const DetectionList& detections, const cv::Mat& image) = 0;
    virtual void Reset() = 0;

    [[nodiscard]] virtual bool SupportsObb() const noexcept = 0;
    [[nodiscard]] virtual bool SupportsReId() const noexcept = 0;
};

}  // namespace boxmot::trackers::base
#pragma once

#include "boxmot/trackers/base/timing.hpp"

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

    // Positive dt is elapsed seconds / reference_dt_s; zero keeps fixed steps.
    // Age, confirmation, and expiry counters always count Update calls.
    virtual TrackOutputList Update(const DetectionList& detections, const cv::Mat& image, double dt = 0.0) = 0;
    virtual void Reset() = 0;

    [[nodiscard]] virtual bool SupportsObb() const noexcept = 0;
    [[nodiscard]] virtual bool SupportsEmbeddings() const noexcept = 0;
};

}  // namespace boxmot::trackers::base

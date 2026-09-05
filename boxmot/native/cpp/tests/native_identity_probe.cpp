#include "botsort/track.hpp"
#include "botsort/types.hpp"
#include "bytetrack/track.hpp"
#include "bytetrack/tracker.hpp"
#include "bytetrack/types.hpp"
#include "occluboost/track.hpp"
#include "occluboost/types.hpp"
#include "ocsort/tracker.hpp"
#include "ocsort/types.hpp"
#include "sfsort/tracker.hpp"
#include "sfsort/types.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <typename Value>
constexpr bool kIsInt64 =
    std::is_same_v<std::remove_cv_t<std::remove_reference_t<Value>>, std::int64_t>;

static_assert(kIsInt64<decltype((std::declval<botsort::Detection&>().cls))>);
static_assert(kIsInt64<decltype((std::declval<botsort::Detection&>().det_ind))>);
static_assert(kIsInt64<decltype((std::declval<botsort::Track&>().id))>);
static_assert(kIsInt64<decltype((std::declval<botsort::TrackOutput&>().id))>);

static_assert(kIsInt64<decltype((std::declval<bytetrack::Detection&>().cls))>);
static_assert(kIsInt64<decltype((std::declval<bytetrack::Detection&>().det_ind))>);
static_assert(kIsInt64<decltype((std::declval<bytetrack::Track&>().id))>);
static_assert(kIsInt64<decltype((std::declval<bytetrack::TrackOutput&>().id))>);

static_assert(kIsInt64<decltype((std::declval<occluboost::Detection&>().cls))>);
static_assert(kIsInt64<decltype((std::declval<occluboost::Detection&>().det_ind))>);
static_assert(kIsInt64<decltype((std::declval<occluboost::KalmanBoxTracker&>().id))>);
static_assert(kIsInt64<decltype((std::declval<occluboost::TrackOutput&>().id))>);

static_assert(kIsInt64<decltype((std::declval<ocsort::Detection&>().cls))>);
static_assert(kIsInt64<decltype((std::declval<ocsort::Detection&>().det_ind))>);
static_assert(
    kIsInt64<decltype((std::declval<ocsort::OCSORTTracker::KalmanBoxTracker&>().id))>);
static_assert(kIsInt64<decltype((std::declval<ocsort::TrackOutput&>().id))>);

static_assert(kIsInt64<decltype((std::declval<sfsort::Detection&>().cls))>);
static_assert(kIsInt64<decltype((std::declval<sfsort::Detection&>().det_ind))>);
static_assert(kIsInt64<decltype((std::declval<sfsort::SFSORTTracker::TrackData&>().track_id))>);
static_assert(kIsInt64<decltype((std::declval<sfsort::TrackOutput&>().id))>);

bytetrack::Detection MakeDetection(const double x,
                                   const std::int64_t class_id,
                                   const std::int64_t detection_index) {
    bytetrack::Detection detection;
    detection.xyxy << x, 0.0, x + 10.0, 20.0;
    detection.conf = 0.95F;
    detection.cls = class_id;
    detection.det_ind = detection_index;
    return detection;
}

bool HasTrack(const std::vector<bytetrack::TrackOutput>& tracks,
              const std::int64_t track_id,
              const std::int64_t class_id,
              const std::int64_t detection_index) {
    return std::any_of(tracks.begin(), tracks.end(), [&](const bytetrack::TrackOutput& track) {
        return track.id == track_id && track.cls == class_id &&
               track.det_ind == detection_index;
    });
}

}  // namespace

int main() {
    bytetrack::Config config;
    config.min_conf = 0.1F;
    config.track_thresh = 0.5F;
    config.match_thresh = 0.8F;

    const std::int64_t first_class = (std::int64_t{1} << 40) + 11;
    const std::int64_t first_index = (std::int64_t{1} << 41) + 17;
    const std::int64_t second_class = first_class + 1;
    const std::int64_t second_index = first_index + 1;
    const bytetrack::Detection first = MakeDetection(0.0, first_class, first_index);
    const bytetrack::Detection second = MakeDetection(100.0, second_class, second_index);

    bytetrack::ByteTrackTracker first_tracker(config);
    const auto first_frame = first_tracker.Update({first}, {});
    if (first_frame.size() != 1 ||
        !HasTrack(first_frame, 1, first_class, first_index)) {
        std::cerr << "first tracker did not preserve int64 metadata\n";
        return 1;
    }

    // Constructing another tracker must not reset or share the first
    // tracker's ID allocator.
    bytetrack::ByteTrackTracker second_tracker(config);
    first_tracker.Update({first, second}, {});
    const auto third_frame = first_tracker.Update({first, second}, {});
    if (third_frame.size() != 2 ||
        !HasTrack(third_frame, 1, first_class, first_index) ||
        !HasTrack(third_frame, 2, second_class, second_index)) {
        std::cerr << "tracker instances share an ID allocator\n";
        return 2;
    }

    const auto independent_frame = second_tracker.Update({second}, {});
    if (independent_frame.size() != 1 || independent_frame.front().id != 1) {
        std::cerr << "independent tracker did not start its own ID sequence\n";
        return 3;
    }

    first_tracker.Reset();
    const auto reset_frame = first_tracker.Update({first}, {});
    if (reset_frame.size() != 1 || reset_frame.front().id != 1) {
        std::cerr << "tracker reset did not reset its own ID sequence\n";
        return 4;
    }

    return 0;
}

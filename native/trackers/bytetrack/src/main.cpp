#include "bytetrack/data_io.hpp"
#include "bytetrack/tracker.hpp"

#include "boxmot/trackers/base/replay.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using bytetrack::ReplayOptions;

ReplayOptions ParseArgs(const int argc, char** argv) {
    const auto args = boxmot::trackers::base::ParseKeyValueArgs(
        argc,
        argv,
        "Usage: bytetrack_replay --mot-root <path> --det-emb-root <path> --detector-name <name>\n"
        "       --sequence <name> --output <path> [options]\n"
    );

    ReplayOptions options;
    options.mot_root = args.at("mot-root");
    options.det_emb_root = args.at("det-emb-root");
    options.detector_name = args.at("detector-name");
    options.sequence = args.at("sequence");
    options.output_path = args.at("output");
    options.conf_threshold = args.count("conf-threshold") ? std::stof(args.at("conf-threshold")) : 0.0F;
    options.target_fps = args.count("target-fps") ? std::stoi(args.at("target-fps")) : 0;
    options.tracker.min_conf = args.count("min-conf") ? std::stof(args.at("min-conf")) : 0.1F;
    options.tracker.track_thresh = args.count("track-thresh") ? std::stof(args.at("track-thresh")) : 0.6F;
    options.tracker.track_buffer = args.count("track-buffer") ? std::stoi(args.at("track-buffer")) : 30;
    options.tracker.match_thresh = args.count("match-thresh") ? std::stof(args.at("match-thresh")) : 0.9F;
    options.tracker.frame_rate = args.count("frame-rate") ? std::stoi(args.at("frame-rate")) : 30;
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const ReplayOptions options = ParseArgs(argc, argv);
        return boxmot::trackers::base::RunTrackerReplay<bytetrack::ByteTrackTracker, bytetrack::ReplaySummary>(
            options,
            bytetrack::LoadSequence,
            bytetrack::SliceDetectionsForFrame,
            bytetrack::ReadImage,
            bytetrack::WriteResultLine
        );
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return 1;
    }
}

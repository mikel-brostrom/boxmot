#include "sfsort/data_io.hpp"
#include "sfsort/tracker.hpp"

#include "boxmot/trackers/base/replay.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using sfsort::ReplayOptions;

ReplayOptions ParseArgs(const int argc, char** argv) {
    const auto args = boxmot::trackers::base::ParseKeyValueArgs(
        argc,
        argv,
        "Usage: sfsort_replay --mot-root <path> --det-emb-root <path> --detector-name <name>\n"
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
    options.tracker.high_th = args.count("high-th") ? std::stof(args.at("high-th")) : 0.6F;
    options.tracker.match_th_first = args.count("match-th-first") ? std::stof(args.at("match-th-first")) : 0.67F;
    options.tracker.new_track_th = args.count("new-track-th") ? std::stof(args.at("new-track-th")) : 0.7F;
    options.tracker.low_th = args.count("low-th") ? std::stof(args.at("low-th")) : 0.1F;
    options.tracker.match_th_second = args.count("match-th-second") ? std::stof(args.at("match-th-second")) : 0.3F;
    options.tracker.dynamic_tuning = args.count("dynamic-tuning") ? std::stoi(args.at("dynamic-tuning")) != 0 : false;
    options.tracker.cth = args.count("cth") ? std::stof(args.at("cth")) : 0.5F;
    options.tracker.high_th_m = args.count("high-th-m") ? std::stof(args.at("high-th-m")) : 0.0F;
    options.tracker.new_track_th_m = args.count("new-track-th-m") ? std::stof(args.at("new-track-th-m")) : 0.0F;
    options.tracker.match_th_first_m = args.count("match-th-first-m") ? std::stof(args.at("match-th-first-m")) : 0.0F;
    options.tracker.obb_theta_damping = args.count("obb-theta-damping") ? std::stof(args.at("obb-theta-damping")) : 0.8F;
    options.tracker.marginal_timeout = args.count("marginal-timeout") ? std::stoi(args.at("marginal-timeout")) : 0;
    options.tracker.central_timeout = args.count("central-timeout") ? std::stoi(args.at("central-timeout")) : 0;
    options.tracker.frame_width = args.count("frame-width") ? std::stoi(args.at("frame-width")) : 0;
    options.tracker.frame_height = args.count("frame-height") ? std::stoi(args.at("frame-height")) : 0;
    options.tracker.horizontal_margin = args.count("horizontal-margin") ? std::stoi(args.at("horizontal-margin")) : 0;
    options.tracker.vertical_margin = args.count("vertical-margin") ? std::stoi(args.at("vertical-margin")) : 0;
    options.tracker.frame_rate = args.count("frame-rate") ? std::stoi(args.at("frame-rate")) : 30;
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const ReplayOptions options = ParseArgs(argc, argv);
        return boxmot::trackers::base::RunTrackerReplay<sfsort::SFSORTTracker, sfsort::ReplaySummary>(
            options,
            sfsort::LoadSequence,
            sfsort::SliceDetectionsForFrame,
            sfsort::ReadImage,
            sfsort::WriteResultLine
        );
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return 1;
    }
}

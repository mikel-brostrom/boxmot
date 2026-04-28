#include "botsort/data_io.hpp"
#include "botsort/tracker.hpp"

#include "boxmot/trackers/base/replay.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using botsort::ReplayOptions;

bool ParseBool(const std::string& value) {
    return value == "1" || value == "true" || value == "True" || value == "TRUE";
}

ReplayOptions ParseArgs(const int argc, char** argv) {
    const auto args = boxmot::trackers::base::ParseKeyValueArgs(
        argc,
        argv,
        "Usage: botsort_replay --mot-root <path> --det-emb-root <path> --detector-name <name>\n"
        "       --reid-name <name> --sequence <name> --output <path> [options]\n"
    );

    ReplayOptions options;
    options.mot_root = args.at("mot-root");
    options.det_emb_root = args.at("det-emb-root");
    options.detector_name = args.at("detector-name");
    options.reid_name = args.at("reid-name");
    options.reid_model_path = args.count("reid-model") ? botsort::fs::path(args.at("reid-model")) : botsort::fs::path();
    options.reid_preprocess = args.count("reid-preprocess") ? args.at("reid-preprocess") : "resize";
    options.sequence = args.at("sequence");
    options.output_path = args.at("output");
    options.conf_threshold = args.count("conf-threshold") ? std::stof(args.at("conf-threshold")) : 0.0F;
    options.target_fps = args.count("target-fps") ? std::stoi(args.at("target-fps")) : 0;
    options.tracker.track_high_thresh = args.count("track-high-thresh") ? std::stof(args.at("track-high-thresh")) : 0.6F;
    options.tracker.track_low_thresh = args.count("track-low-thresh") ? std::stof(args.at("track-low-thresh")) : 0.1F;
    options.tracker.new_track_thresh = args.count("new-track-thresh") ? std::stof(args.at("new-track-thresh")) : 0.7F;
    options.tracker.track_buffer = args.count("track-buffer") ? std::stoi(args.at("track-buffer")) : 30;
    options.tracker.match_thresh = args.count("match-thresh") ? std::stof(args.at("match-thresh")) : 0.8F;
    options.tracker.proximity_thresh = args.count("proximity-thresh") ? std::stof(args.at("proximity-thresh")) : 0.5F;
    options.tracker.appearance_thresh = args.count("appearance-thresh") ? std::stof(args.at("appearance-thresh")) : 0.25F;
    options.tracker.cmc_method = args.count("cmc-method") ? args.at("cmc-method") : "ecc";
    options.tracker.frame_rate = args.count("frame-rate") ? std::stoi(args.at("frame-rate")) : 30;
    options.tracker.fuse_first_associate = args.count("fuse-first-associate") && ParseBool(args.at("fuse-first-associate"));
    options.tracker.with_reid = !args.count("with-reid") || ParseBool(args.at("with-reid"));
    options.tracker.reid_model_path = options.reid_model_path.string();
    options.tracker.reid_preprocess = options.reid_preprocess;
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const ReplayOptions options = ParseArgs(argc, argv);
        return boxmot::trackers::base::RunTrackerReplay<botsort::BotSortTracker, botsort::ReplaySummary>(
            options,
            botsort::LoadSequence,
            botsort::SliceDetectionsForFrame,
            botsort::ReadImage,
            botsort::WriteMotLine
        );
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return 1;
    }
}

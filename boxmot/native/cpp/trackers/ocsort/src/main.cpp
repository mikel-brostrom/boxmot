#include "ocsort/data_io.hpp"
#include "ocsort/tracker.hpp"

#include "boxmot/trackers/base/replay.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using ocsort::ReplayOptions;

ReplayOptions ParseArgs(const int argc, char** argv) {
    const auto args = boxmot::trackers::base::ParseKeyValueArgs(
        argc,
        argv,
        "Usage: ocsort_replay --mot-root <path> --det-emb-root <path> --detector-name <name>\n"
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
    options.tracker.det_thresh = args.count("det-thresh") ? std::stof(args.at("det-thresh")) : 0.6F;
    options.tracker.iou_threshold = args.count("iou-threshold") ? std::stof(args.at("iou-threshold")) : 0.3F;
    options.tracker.max_age = args.count("max-age") ? std::stoi(args.at("max-age")) : 30;
    options.tracker.min_hits = args.count("min-hits") ? std::stoi(args.at("min-hits")) : 3;
    options.tracker.delta_t = args.count("delta-t") ? std::stoi(args.at("delta-t")) : 3;
    options.tracker.use_byte = args.count("use-byte") ? std::stoi(args.at("use-byte")) != 0 : false;
    options.tracker.inertia = args.count("inertia") ? std::stof(args.at("inertia")) : 0.1F;
    options.tracker.q_xy_scaling = args.count("q-xy-scaling") ? std::stof(args.at("q-xy-scaling")) : 0.01F;
    options.tracker.q_s_scaling = args.count("q-s-scaling") ? std::stof(args.at("q-s-scaling")) : 0.0001F;
    options.tracker.max_obs = args.count("max-obs") ? std::stoi(args.at("max-obs")) : 50;
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const ReplayOptions options = ParseArgs(argc, argv);
        return boxmot::trackers::base::RunTrackerReplay<ocsort::OCSORTTracker, ocsort::ReplaySummary>(
            options,
            ocsort::LoadSequence,
            ocsort::SliceDetectionsForFrame,
            ocsort::ReadImage,
            ocsort::WriteResultLine
        );
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return 1;
    }
}

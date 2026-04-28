#include "occluboost/data_io.hpp"
#include "occluboost/tracker.hpp"

#include "boxmot/trackers/base/replay.hpp"

#include <opencv2/core.hpp>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using occluboost::ReplayOptions;

bool ParseBool(const std::string& value) {
    return value == "1" || value == "true" || value == "True" || value == "TRUE";
}

float Get(const std::unordered_map<std::string, std::string>& args, const char* key, const float fallback) {
    const auto it = args.find(key);
    return it == args.end() ? fallback : std::stof(it->second);
}
int GetI(const std::unordered_map<std::string, std::string>& args, const char* key, const int fallback) {
    const auto it = args.find(key);
    return it == args.end() ? fallback : std::stoi(it->second);
}
std::string GetS(const std::unordered_map<std::string, std::string>& args, const char* key, const char* fallback) {
    const auto it = args.find(key);
    return it == args.end() ? std::string(fallback) : it->second;
}
bool GetB(const std::unordered_map<std::string, std::string>& args, const char* key, const bool fallback) {
    const auto it = args.find(key);
    return it == args.end() ? fallback : ParseBool(it->second);
}

ReplayOptions ParseArgs(const int argc, char** argv) {
    const auto args = boxmot::trackers::base::ParseKeyValueArgs(
        argc,
        argv,
        "Usage: occluboost_replay --mot-root <path> --det-emb-root <path> --detector-name <name>\n"
        "       --reid-name <name> --sequence <name> --output <path> [options]\n"
    );

    ReplayOptions options;
    options.mot_root = args.at("mot-root");
    options.det_emb_root = args.at("det-emb-root");
    options.detector_name = args.at("detector-name");
    options.reid_name = args.at("reid-name");
    options.reid_model_path = args.count("reid-model") ? occluboost::fs::path(args.at("reid-model")) : occluboost::fs::path();
    options.reid_preprocess = GetS(args, "reid-preprocess", "resize");
    options.sequence = args.at("sequence");
    options.output_path = args.at("output");
    options.conf_threshold = Get(args, "conf-threshold", 0.0F);
    options.target_fps = GetI(args, "target-fps", 0);

    auto& t = options.tracker;
    t.max_age = GetI(args, "max-age", t.max_age);
    t.min_hits = GetI(args, "min-hits", t.min_hits);
    t.det_thresh = Get(args, "det-thresh", t.det_thresh);
    t.iou_threshold = Get(args, "iou-threshold", t.iou_threshold);
    t.min_box_area = GetI(args, "min-box-area", t.min_box_area);
    t.aspect_ratio_thresh = Get(args, "aspect-ratio-thresh", t.aspect_ratio_thresh);
    t.lambda_iou = Get(args, "lambda-iou", t.lambda_iou);
    t.lambda_mhd = Get(args, "lambda-mhd", t.lambda_mhd);
    t.lambda_shape = Get(args, "lambda-shape", t.lambda_shape);
    t.use_dlo_boost = GetB(args, "use-dlo-boost", t.use_dlo_boost);
    t.use_duo_boost = GetB(args, "use-duo-boost", t.use_duo_boost);
    t.dlo_boost_coef = Get(args, "dlo-boost-coef", t.dlo_boost_coef);
    t.s_sim_corr = GetB(args, "s-sim-corr", t.s_sim_corr);
    t.use_rich_s = GetB(args, "use-rich-s", t.use_rich_s);
    t.use_sb = GetB(args, "use-sb", t.use_sb);
    t.use_vt = GetB(args, "use-vt", t.use_vt);
    t.with_reid = GetB(args, "with-reid", t.with_reid);
    t.cmc_method = GetS(args, "cmc-method", t.cmc_method.c_str());
    t.max_obs = GetI(args, "max-obs", t.max_obs);

    t.recovery_appearance_thresh = Get(args, "recovery-appearance-thresh", t.recovery_appearance_thresh);
    t.recovery_iou_thresh = Get(args, "recovery-iou-thresh", t.recovery_iou_thresh);
    t.recovery_max_age = GetI(args, "recovery-max-age", t.recovery_max_age);
    t.feat_alpha = Get(args, "feat-alpha", t.feat_alpha);
    t.track_low_thresh = Get(args, "track-low-thresh", t.track_low_thresh);
    t.second_iou_thresh = Get(args, "second-iou-thresh", t.second_iou_thresh);
    t.second_appearance_thresh = Get(args, "second-appearance-thresh", t.second_appearance_thresh);
    t.second_pass_max_age = GetI(args, "second-pass-max-age", t.second_pass_max_age);
    t.second_pass_min_hits = GetI(args, "second-pass-min-hits", t.second_pass_min_hits);
    t.use_second_pass = GetB(args, "use-second-pass", t.use_second_pass);
    t.new_track_thresh = Get(args, "new-track-thresh", t.new_track_thresh);
    t.confirm_hits = GetI(args, "confirm-hits", t.confirm_hits);
    t.instant_confirm_thresh = Get(args, "instant-confirm-thresh", t.instant_confirm_thresh);
    t.tentative_max_age = GetI(args, "tentative-max-age", t.tentative_max_age);
    t.duplicate_iou_thresh = Get(args, "duplicate-iou-thresh", t.duplicate_iou_thresh);
    t.ams_enabled = GetB(args, "ams-enabled", t.ams_enabled);
    t.ams_alpha0 = Get(args, "ams-alpha0", t.ams_alpha0);
    t.ams_threshold = Get(args, "ams-threshold", t.ams_threshold);
    t.ams_buffer_size = GetI(args, "ams-buffer-size", t.ams_buffer_size);
    t.ams_shrink_ratio = Get(args, "ams-shrink-ratio", t.ams_shrink_ratio);

    t.reid_model_path = options.reid_model_path.string();
    t.reid_preprocess = options.reid_preprocess;
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    // The replay binary is normally launched as one of N parallel subprocesses
    // by the Python eval pipeline. If OpenCV spawns its default thread pool
    // inside each subprocess we end up with N * cores threads contending for
    // cores, which on macOS manifests as the eval visibly freezing under load.
    // Allow opting out via BOXMOT_NATIVE_CV_THREADS for benchmarking.
    if (const char* env = std::getenv("BOXMOT_NATIVE_CV_THREADS")) {
        cv::setNumThreads(std::atoi(env));
    } else {
        cv::setNumThreads(1);
    }
    try {
        const ReplayOptions options = ParseArgs(argc, argv);
        const bool cmc_disabled = options.tracker.cmc_method.empty() || options.tracker.cmc_method == "none";
        auto read_image = [cmc_disabled](const occluboost::fs::path& path) -> cv::Mat {
            // Skip the JPEG decode entirely when CMC is off (the tracker never reads
            // the pixels). Return a 1x1 stub so the base replay loop does not skip
            // the frame on `image.empty()`.
            if (cmc_disabled) {
                return cv::Mat::ones(1, 1, CV_8UC1);
            }
            return occluboost::ReadImage(path);
        };
        return boxmot::trackers::base::RunTrackerReplay<occluboost::OccluBoostTracker, occluboost::ReplaySummary>(
            options,
            occluboost::LoadSequence,
            occluboost::SliceDetectionsForFrame,
            read_image,
            occluboost::WriteMotLine
        );
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return 1;
    }
}

// BoxMOT DeepStream adapter — YAML configuration parser.
//
// Parses a DeepStream-style YAML config file into BoxMOTTrackerConfig.
// The config format mirrors NvMultiObjectTracker's YAML conventions so
// users familiar with DeepStream tracker configuration feel at home.

#include "deepstream/adapter_types.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace boxmot::deepstream {

namespace {

// Minimal YAML-like parser for flat/nested key-value configs.
// DeepStream tracker configs are simple enough that we don't need a full
// YAML library — they use indentation-based sections with scalar values.

std::string Trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

std::string ToLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool ParseBool(const std::string& val) {
    std::string lower = ToLower(Trim(val));
    return lower == "1" || lower == "true" || lower == "yes" || lower == "on";
}

float ParseFloat(const std::string& val) {
    try { return std::stof(Trim(val)); }
    catch (...) { return 0.0f; }
}

int ParseInt(const std::string& val) {
    try { return std::stoi(Trim(val)); }
    catch (...) { return 0; }
}

std::vector<float> ParseFloatList(const std::string& val) {
    std::vector<float> result;
    std::string cleaned = Trim(val);
    // Remove brackets if present
    if (!cleaned.empty() && (cleaned.front() == '[' || cleaned.front() == '{')) {
        cleaned = cleaned.substr(1);
    }
    if (!cleaned.empty() && (cleaned.back() == ']' || cleaned.back() == '}')) {
        cleaned.pop_back();
    }
    std::stringstream ss(cleaned);
    std::string item;
    while (std::getline(ss, item, ',')) {
        std::string trimmed = Trim(item);
        if (!trimmed.empty()) {
            try { result.push_back(std::stof(trimmed)); }
            catch (...) {}
        }
    }
    return result;
}

std::vector<int> ParseIntList(const std::string& val) {
    std::vector<int> result;
    std::string cleaned = Trim(val);
    if (!cleaned.empty() && (cleaned.front() == '[' || cleaned.front() == '{')) {
        cleaned = cleaned.substr(1);
    }
    if (!cleaned.empty() && (cleaned.back() == ']' || cleaned.back() == '}')) {
        cleaned.pop_back();
    }
    std::stringstream ss(cleaned);
    std::string item;
    while (std::getline(ss, item, ',')) {
        std::string trimmed = Trim(item);
        if (!trimmed.empty()) {
            try { result.push_back(std::stoi(trimmed)); }
            catch (...) {}
        }
    }
    return result;
}

struct ConfigSection {
    std::string name;
    std::unordered_map<std::string, std::string> values;
};

std::vector<ConfigSection> ParseYamlSections(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + filepath);
    }

    std::vector<ConfigSection> sections;
    ConfigSection current;
    current.name = "root";

    std::string line;
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        std::string trimmed = Trim(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;

        // Check if this is a section header (no leading whitespace, ends with ':')
        if (line[0] != ' ' && line[0] != '\t' && trimmed.back() == ':' &&
            trimmed.find(':') == trimmed.size() - 1) {
            if (!current.values.empty() || current.name != "root") {
                sections.push_back(std::move(current));
            }
            current = ConfigSection{};
            current.name = trimmed.substr(0, trimmed.size() - 1);
            continue;
        }

        // Key-value pair
        auto colon_pos = trimmed.find(':');
        if (colon_pos != std::string::npos) {
            std::string key = Trim(trimmed.substr(0, colon_pos));
            std::string value = Trim(trimmed.substr(colon_pos + 1));
            // Remove trailing comments
            auto comment_pos = value.find('#');
            if (comment_pos != std::string::npos) {
                value = Trim(value.substr(0, comment_pos));
            }
            // Remove quotes
            if (value.size() >= 2 &&
                ((value.front() == '"' && value.back() == '"') ||
                 (value.front() == '\'' && value.back() == '\''))) {
                value = value.substr(1, value.size() - 2);
            }
            current.values[key] = value;
        }
    }

    if (!current.values.empty() || current.name != "root") {
        sections.push_back(std::move(current));
    }

    return sections;
}

}  // namespace

BoxMOTTrackerConfig ParseConfig(const std::string& config_path) {
    BoxMOTTrackerConfig config;

    auto sections = ParseYamlSections(config_path);

    for (const auto& section : sections) {
        const auto& vals = section.values;
        const std::string sec_name = ToLower(section.name);

        // Helper to get a value with case-insensitive key lookup
        auto get = [&vals](const std::string& key) -> std::string {
            auto it = vals.find(key);
            if (it != vals.end()) return it->second;
            return "";
        };

        auto has = [&vals](const std::string& key) -> bool {
            return vals.find(key) != vals.end();
        };

        if (sec_name == "root" || sec_name == "baseconfig" || sec_name == "boxmot") {
            // General settings
            if (has("algorithm")) {
                std::string algo = ToLower(get("algorithm"));
                if (algo == "botsort" || algo == "bot_sort") config.algorithm = TrackerAlgorithm::kBotSort;
                else if (algo == "bytetrack" || algo == "byte_track") config.algorithm = TrackerAlgorithm::kByteTrack;
                else if (algo == "ocsort" || algo == "oc_sort") config.algorithm = TrackerAlgorithm::kOcSort;
                else if (algo == "sfsort" || algo == "sf_sort") config.algorithm = TrackerAlgorithm::kSfSort;
                else if (algo == "occluboost" || algo == "occlu_boost") config.algorithm = TrackerAlgorithm::kOccluBoost;
            }
            if (has("trackHighThresh")) config.track_high_thresh = ParseFloat(get("trackHighThresh"));
            if (has("trackLowThresh")) config.track_low_thresh = ParseFloat(get("trackLowThresh"));
            if (has("newTrackThresh")) config.new_track_thresh = ParseFloat(get("newTrackThresh"));
            if (has("trackBuffer")) config.track_buffer = ParseInt(get("trackBuffer"));
            if (has("matchThresh")) config.match_thresh = ParseFloat(get("matchThresh"));
            if (has("frameRate")) config.frame_rate = ParseInt(get("frameRate"));
            if (has("maxTargetsPerStream")) config.max_targets_per_stream = ParseInt(get("maxTargetsPerStream"));
            if (has("withReId")) config.with_reid = ParseBool(get("withReId"));
            if (has("enableReId")) config.enable_reid = ParseBool(get("enableReId"));

            // BoTSORT specific
            if (has("proximityThresh")) config.proximity_thresh = ParseFloat(get("proximityThresh"));
            if (has("appearanceThresh")) config.appearance_thresh = ParseFloat(get("appearanceThresh"));
            if (has("cmcMethod")) config.cmc_method = get("cmcMethod");
            if (has("fuseFirstAssociate")) config.fuse_first_associate = ParseBool(get("fuseFirstAssociate"));
            if (has("maxObs")) config.max_obs = ParseInt(get("maxObs"));

            // OCSORT specific
            if (has("deltaT")) config.delta_t = ParseFloat(get("deltaT"));
            if (has("iouThresh")) config.iou_thresh = ParseFloat(get("iouThresh"));
            if (has("velDirWeight")) config.vel_dir_weight = ParseFloat(get("velDirWeight"));
        }

        if (sec_name == "targetmanagement") {
            if (has("probationAge")) config.probation_age = ParseInt(get("probationAge"));
            if (has("maxShadowTrackingAge")) config.max_shadow_tracking_age = ParseInt(get("maxShadowTrackingAge"));
            if (has("earlyTerminationAge")) config.early_termination_age = ParseInt(get("earlyTerminationAge"));
            if (has("maxTargetsPerStream")) config.max_targets_per_stream = ParseInt(get("maxTargetsPerStream"));
            if (has("outputTerminatedTracks")) config.output_terminated_tracks = ParseBool(get("outputTerminatedTracks"));
            if (has("outputShadowTracks")) config.output_shadow_tracks = ParseBool(get("outputShadowTracks"));
            if (has("supportPastFrame")) config.support_past_frame = ParseBool(get("supportPastFrame"));
        }

        if (sec_name == "dataassociator") {
            if (has("associationMatcherType")) config.association_matcher_type = ParseInt(get("associationMatcherType"));
            if (has("checkClassMatch")) config.check_class_match = ParseBool(get("checkClassMatch"));
            if (has("minMatchingScore4Overall")) config.min_matching_score_overall = ParseFloat(get("minMatchingScore4Overall"));
            if (has("minMatchingScore4Iou")) config.min_matching_score_iou = ParseFloat(get("minMatchingScore4Iou"));
            if (has("matchingScoreWeight4Iou")) config.matching_score_weight_iou = ParseFloat(get("matchingScoreWeight4Iou"));
            if (has("matchingScoreWeight4ReIDSimilarity")) config.matching_score_weight_reid = ParseFloat(get("matchingScoreWeight4ReIDSimilarity"));
            if (has("minIouDiff4NewTarget")) config.min_iou_diff_new_target = ParseFloat(get("minIouDiff4NewTarget"));
        }

        if (sec_name == "reid") {
            // TensorRT ReID configuration (matching DeepStream's ReID section)
            if (has("onnxFile")) config.reid.onnx_file = get("onnxFile");
            if (has("tltEncodedModel")) config.reid.tlt_encoded_model = get("tltEncodedModel");
            if (has("tltModelKey")) config.reid.tlt_model_key = get("tltModelKey");
            if (has("modelEngineFile")) config.reid.model_engine_file = get("modelEngineFile");
            if (has("calibrationTableFile")) config.reid.calibration_table_file = get("calibrationTableFile");

            if (has("batchSize")) config.reid.batch_size = ParseInt(get("batchSize"));
            if (has("networkMode")) config.reid.network_mode = ParseInt(get("networkMode"));
            if (has("workspaceSize")) config.reid.workspace_size = ParseInt(get("workspaceSize"));
            if (has("inferDims")) config.reid.infer_dims = ParseIntList(get("inferDims"));
            if (has("inputOrder")) config.reid.input_order = ParseInt(get("inputOrder"));
            if (has("colorFormat")) config.reid.color_format = ParseInt(get("colorFormat"));

            if (has("netScaleFactor")) config.reid.net_scale_factor = ParseFloat(get("netScaleFactor"));
            if (has("offsets")) config.reid.offsets = ParseFloatList(get("offsets"));

            if (has("reidFeatureSize")) config.reid.reid_feature_size = ParseInt(get("reidFeatureSize"));
            if (has("reidHistorySize")) config.reid.reid_history_size = ParseInt(get("reidHistorySize"));
            if (has("addFeatureNormalization")) config.reid.add_feature_normalization = ParseBool(get("addFeatureNormalization"));
            if (has("keepAspc")) config.reid.keep_aspect_ratio = ParseBool(get("keepAspc"));

            // reidType: 0=DUMMY, 1=NvDEEPSORT, 2=ReAssoc, 3=Both
            if (has("reidType")) {
                int reid_type = ParseInt(get("reidType"));
                config.enable_reid = (reid_type > 0);
            }
        }
    }

    return config;
}

}  // namespace boxmot::deepstream

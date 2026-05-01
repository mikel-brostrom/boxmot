#pragma once

#include <opencv2/core.hpp>

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace boxmot::trackers::base {

inline std::string JsonEscape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char ch : value) {
        if (ch == '\\' || ch == '"') {
            escaped.push_back('\\');
        }
        escaped.push_back(ch);
    }
    return escaped;
}

template <typename ReplaySummaryT>
void PrintSummaryJson(const ReplaySummaryT& summary, std::ostream& stream = std::cout) {
    stream << "{\"sequence\":\"" << JsonEscape(summary.sequence) << "\","
           << "\"num_frames\":" << summary.num_frames << ','
           << "\"track_time_ms\":" << std::fixed << std::setprecision(3) << summary.track_time_ms << ','
           << "\"kept_frame_ids\":[";
    for (std::size_t index = 0; index < summary.kept_frame_ids.size(); ++index) {
        if (index > 0) {
            stream << ',';
        }
        stream << summary.kept_frame_ids[index];
    }
    stream << "]}\n";
}

inline void PrintProgress(
    const std::string& sequence,
    const std::size_t current,
    const std::size_t total,
    std::ostream& stream = std::cerr
) {
    stream << "BOXMOT_PROGRESS\t"
           << sequence << '\t'
           << current << '\t'
           << total << std::endl;
}

inline std::unordered_map<std::string, std::string> ParseKeyValueArgs(
    const int argc,
    char** argv,
    const std::string_view usage,
    std::ostream& stream = std::cerr
) {
    auto print_usage = [&]() {
        stream << std::string(usage);
    };

    if (argc <= 1) {
        print_usage();
        throw std::runtime_error("Missing arguments");
    }

    std::unordered_map<std::string, std::string> args;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--help" || key == "-h") {
            print_usage();
            std::exit(0);
        }
        if (key.rfind("--", 0) != 0 || index + 1 >= argc) {
            print_usage();
            throw std::runtime_error("Invalid arguments");
        }
        args[key.substr(2)] = argv[++index];
    }
    return args;
}

template <
    typename Tracker,
    typename ReplaySummary,
    typename Options,
    typename LoadSequence,
    typename SliceDetections,
    typename ReadImage,
    typename WriteTrack>
int RunTrackerReplay(
    const Options& options,
    LoadSequence load_sequence,
    SliceDetections slice_detections,
    ReadImage read_image,
    WriteTrack write_track
) {
    const auto sequence = load_sequence(options);
    Tracker tracker(options.tracker);

    const std::filesystem::path output_dir = options.output_path.parent_path();
    if (!output_dir.empty()) {
        std::filesystem::create_directories(output_dir);
    }
    std::ofstream output(options.output_path);
    if (!output) {
        throw std::runtime_error("Failed to open output file: " + options.output_path.string());
    }

    ReplaySummary summary;
    summary.sequence = sequence.name;

    std::size_t row_offset = 0;
    const std::size_t total_frames = sequence.frame_ids.size();
    for (std::size_t index = 0; index < sequence.frame_ids.size(); ++index) {
        const int frame_id = sequence.frame_ids[index];
        cv::Mat image = read_image(sequence.frame_paths[index]);
        if (image.empty()) {
            PrintProgress(sequence.name, index + 1, total_frames);
            continue;
        }

        const auto frame_detections = slice_detections(sequence, frame_id, row_offset, options.conf_threshold);

        const auto started = std::chrono::steady_clock::now();
        const auto tracks = tracker.Update(frame_detections, image);
        const auto ended = std::chrono::steady_clock::now();

        summary.track_time_ms += std::chrono::duration<double, std::milli>(ended - started).count();
        summary.kept_frame_ids.push_back(frame_id);
        ++summary.num_frames;

        for (const auto& track : tracks) {
            write_track(output, frame_id, track);
        }

        PrintProgress(sequence.name, index + 1, total_frames);
    }

    PrintSummaryJson(summary);
    return 0;
}

}  // namespace boxmot::trackers::base

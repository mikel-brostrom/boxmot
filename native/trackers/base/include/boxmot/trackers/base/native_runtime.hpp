#pragma once

#include <opencv2/core.hpp>

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace boxmot::trackers::base {

inline void SetLastError(std::string& last_error, std::string_view message) {
    last_error.assign(message.data(), message.size());
}

template <typename Fn>
int GuardCall(Fn&& fn, std::string& last_error, std::string_view unknown_message) {
    try {
        fn();
        last_error.clear();
        return 1;
    } catch (const std::exception& exc) {
        SetLastError(last_error, exc.what());
        return 0;
    } catch (...) {
        SetLastError(last_error, unknown_message);
        return 0;
    }
}

inline int CvImageType(const int channels, std::string_view unsupported_message) {
    if (channels == 1) {
        return CV_8UC1;
    }
    if (channels == 3) {
        return CV_8UC3;
    }
    if (channels == 4) {
        return CV_8UC4;
    }
    throw std::runtime_error(std::string(unsupported_message));
}

}  // namespace boxmot::trackers::base

#include BOXMOT_CMC_HEADER

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace tracker = BOXMOT_CMC_NAMESPACE;

namespace {

cv::Mat Translate(const cv::Mat& image, const double dx, const double dy) {
    const cv::Mat transform = (cv::Mat_<float>(2, 3) << 1.0F, 0.0F, dx, 0.0F, 1.0F, dy);
    cv::Mat translated;
    cv::warpAffine(image, translated, transform, image.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    return translated;
}

void CheckTranslation(const cv::Mat& warp, const double dx, const double dy) {
    if (std::abs(warp.at<float>(0, 2) - dx) > 1.5 || std::abs(warp.at<float>(1, 2) - dy) > 1.5) {
        throw std::runtime_error("Camera translation was not recovered in original image coordinates.");
    }
}

void CheckIdentity(const cv::Mat& warp) {
    if (cv::norm(warp, cv::Mat::eye(2, 3, CV_32F), cv::NORM_INF) != 0.0) {
        throw std::runtime_error("A change in input resolution must reset camera motion history.");
    }
}

}  // namespace

int main(const int argc, char* argv[]) {
    if (argc != 2) {
        return 2;
    }
    auto cmc = tracker::CreateCameraMotionCompensator(argv[1]);
    cv::RNG random(42);
    cv::Mat texture(36, 64, CV_8UC3);
    random.fill(texture, cv::RNG::UNIFORM, 0, 255);
    cv::Mat image;
    cv::resize(texture, image, cv::Size(640, 360), 0.0, 0.0, cv::INTER_CUBIC);
    CheckIdentity(cmc->Apply(image, {}));
    for (int frame = 1; frame <= 8; ++frame) {
        CheckTranslation(cmc->Apply(Translate(image, 4.0 * frame, 2.0 * frame), {}), 4.0, 2.0);
    }

    // Both resolutions produce the same estimator dimensions. History must
    // still reset because the transform's original coordinate system changed.
    cv::Mat larger;
    cv::resize(image, larger, cv::Size(1280, 720), 0.0, 0.0, cv::INTER_LINEAR);
    CheckIdentity(cmc->Apply(larger, {}));
    CheckTranslation(cmc->Apply(Translate(larger, 8.0, 4.0), {}), 8.0, 4.0);

    if (std::string(argv[1]) == "sof") {
        tracker::Detection foreground;
        foreground.xyxy << 0.0, 0.0, larger.cols, larger.rows;
        CheckIdentity(cmc->Apply(Translate(larger, 16.0, 8.0), {foreground}));
        CheckIdentity(cmc->Apply(larger, {}));
        CheckTranslation(cmc->Apply(Translate(larger, 8.0, 4.0), {}), 8.0, 4.0);
    }

    // A changed aspect ratio also changes the working shape; do not pass
    // incompatible image pyramids into optical flow or ECC.
    CheckIdentity(cmc->Apply(image(cv::Rect(0, 0, 400, 360)), {}));

    // Grayscale frames below both working widths bypass resize and conversion.
    // A recycled capture buffer must still preserve independent frame history.
    cv::Mat grayscale;
    cv::cvtColor(texture, grayscale, cv::COLOR_BGR2GRAY);
    cv::resize(grayscale, grayscale, cv::Size(96, 54), 0.0, 0.0, cv::INTER_CUBIC);
    cv::GaussianBlur(grayscale, grayscale, cv::Size(5, 5), 0.0);
    const cv::Mat next_grayscale = Translate(grayscale, 2.0, 1.0);
    auto copied = tracker::CreateCameraMotionCompensator(argv[1]);
    auto recycled = tracker::CreateCameraMotionCompensator(argv[1]);
    CheckIdentity(copied->Apply(grayscale.clone(), {}));
    CheckIdentity(recycled->Apply(grayscale, {}));
    next_grayscale.copyTo(grayscale);
    const cv::Mat expected = copied->Apply(next_grayscale.clone(), {});
    const cv::Mat actual = recycled->Apply(grayscale, {});
    CheckTranslation(expected, 2.0, 1.0);
    if (cv::norm(actual, expected, cv::NORM_INF) > 1.0e-4) {
        throw std::runtime_error("Reusing a grayscale capture buffer changed camera motion history.");
    }
    std::cout << "Camera motion translation, resolution recovery and buffer reuse passed.\n";
    return 0;
}

#pragma once

#include "boxmot/trackers/base/assignment.hpp"
#include "boxmot/trackers/base/base_tracker.hpp"
#include "ocsort/kalman_filter.hpp"
#include "ocsort/types.hpp"

#include <opencv2/core.hpp>

#include <deque>
#include <optional>
#include <unordered_map>
#include <vector>

namespace ocsort {

class OCSORTTracker final : public boxmot::trackers::base::TrackerBase<Detection, TrackOutput> {
public:
    struct KalmanBoxTracker {
        static int count;

        explicit KalmanBoxTracker(
            const Detection& detection,
            int delta_t,
            int max_obs,
            double q_xy_scaling,
            double q_s_scaling,
            bool is_obb
        );

        [[nodiscard]] Eigen::VectorXd Predict();
        void Update(const Detection* detection);
        [[nodiscard]] Eigen::VectorXd GetState() const;
        [[nodiscard]] Eigen::VectorXd CurrentOutputBox() const;

        static void ResetCount();

        int det_ind = -1;
        double q_xy_scaling = 0.01;
        double q_s_scaling = 0.0001;
        double q_a_scaling = 0.0001;
        bool is_obb = false;
        KalmanFilterXYSR kf;
        int time_since_update = 0;
        int id = 0;
        int max_obs = 50;
        int hits = 0;
        int hit_streak = 0;
        int age = 0;
        float conf = 0.0F;
        int cls = 0;
        Eigen::VectorXd last_observation;
        std::unordered_map<int, Eigen::VectorXd> observations;
        std::deque<Eigen::VectorXd> history_observations;
        std::optional<Eigen::Vector2d> velocity;
        int delta_t = 3;

    private:
        [[nodiscard]] static Eigen::Vector4d XyxyToXysr(const Eigen::Vector4d& bbox);
        [[nodiscard]] static Eigen::Matrix<double, 5, 1> ConvertObbToZ(const Eigen::Matrix<double, 5, 1>& obb);
        [[nodiscard]] static Eigen::VectorXd ConvertXToBbox(const KalmanFilterXYSR::Vector& state);
        [[nodiscard]] static Eigen::VectorXd ConvertXToObb(const KalmanFilterXYSR::Vector& state);
        [[nodiscard]] static Eigen::Vector2d SpeedDirection(const Eigen::VectorXd& bbox1, const Eigen::VectorXd& bbox2, bool is_obb_mode);
        [[nodiscard]] static Eigen::VectorXd ObservationVector(const Detection& detection);
    };

    explicit OCSORTTracker(Config config);

    std::vector<TrackOutput> Update(const std::vector<Detection>& detections, const cv::Mat& image) override;
    void Reset() override;

    [[nodiscard]] bool SupportsObb() const noexcept override { return true; }
    [[nodiscard]] bool SupportsReId() const noexcept override { return false; }

private:
    using AssignmentResult = boxmot::trackers::base::AssignmentResult;

    [[nodiscard]] static Eigen::VectorXd PlaceholderObservation(bool is_obb_mode);
    [[nodiscard]] static Eigen::VectorXd KPreviousObs(
        const std::unordered_map<int, Eigen::VectorXd>& observations,
        int current_age,
        int k,
        bool is_obb_mode
    );
    [[nodiscard]] static Eigen::MatrixXd SimilarityMatrix(
        const std::vector<Eigen::VectorXd>& detections,
        const std::vector<Eigen::VectorXd>& tracks,
        bool is_obb_mode
    );
    [[nodiscard]] static Eigen::MatrixXd DirectionCost(
        const std::vector<Eigen::VectorXd>& detections,
        const std::vector<Eigen::VectorXd>& previous_obs,
        const std::vector<Eigen::Vector2d>& velocities,
        const std::vector<float>& scores,
        float inertia,
        bool is_obb_mode
    );
    [[nodiscard]] static AssignmentResult Associate(
        const std::vector<Eigen::VectorXd>& detections,
        const std::vector<Eigen::VectorXd>& trackers,
        const std::vector<Eigen::Vector2d>& velocities,
        const std::vector<Eigen::VectorXd>& previous_obs,
        float iou_threshold,
        float inertia,
        bool is_obb_mode
    );
    [[nodiscard]] static TrackOutput FormatTrack(const KalmanBoxTracker& track);

    Config config_;
    int frame_count_ = 0;
    bool detection_mode_ready_ = false;
    bool is_obb_mode_ = false;
    std::vector<KalmanBoxTracker> active_tracks_;
};

}  // namespace ocsort

#pragma once

#include <Eigen/Dense>

#include <deque>
#include <optional>

namespace ocsort {

class KalmanFilterXYSR {
public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;

    KalmanFilterXYSR(int dim_x = 7, int dim_z = 4, int max_obs = 50);

    [[nodiscard]] int dim_x() const noexcept { return dim_x_; }
    [[nodiscard]] int dim_z() const noexcept { return dim_z_; }
    [[nodiscard]] bool is_obb() const noexcept { return is_obb_; }

    void Predict();
    void Update(const Vector& measurement);
    void UpdateMissing();

    Vector x;
    Matrix P;
    Matrix F;
    Matrix H;
    Matrix Q;
    Matrix R;

private:
    struct SavedState {
        Vector x;
        Matrix P;
        std::deque<std::optional<Vector>> history_obs;
        std::optional<Vector> last_measurement;
        bool observed = false;
    };

    [[nodiscard]] static double WrapAngle(double angle);
    [[nodiscard]] Vector AlignObbMeasurement(const Vector& measurement) const;
    void AppendHistory(std::optional<Vector> measurement);
    void EnforceStateConstraints();
    void Freeze();
    void Unfreeze();

    int dim_x_ = 7;
    int dim_z_ = 4;
    bool is_obb_ = false;
    int max_obs_ = 50;
    std::deque<std::optional<Vector>> history_obs_;
    std::optional<SavedState> saved_state_;
    std::optional<Vector> last_measurement_;
    bool observed_ = false;
};

}  // namespace ocsort
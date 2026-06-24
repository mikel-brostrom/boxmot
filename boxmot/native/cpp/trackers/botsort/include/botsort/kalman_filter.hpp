#pragma once

#include <Eigen/Dense>

namespace botsort {

class KalmanFilterXYWH {
public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;

    explicit KalmanFilterXYWH(int ndim = 4);

    [[nodiscard]] int ndim() const { return ndim_; }
    [[nodiscard]] int dim_x() const { return dim_x_; }

    std::pair<Vector, Matrix> Initiate(const Vector& measurement) const;
    std::pair<Vector, Matrix> Predict(const Vector& mean, const Matrix& covariance) const;
    std::pair<Vector, Matrix> Update(
        const Vector& mean,
        const Matrix& covariance,
        const Vector& measurement,
        float confidence = 0.0F
    ) const;

private:
    Vector InitialCovarianceStd(const Vector& measurement) const;
    std::pair<Vector, Vector> ProcessNoiseStd(const Vector& mean) const;
    Vector MeasurementNoiseStd(const Vector& mean, float confidence) const;
    std::pair<Vector, Matrix> Project(const Vector& mean, const Matrix& covariance, float confidence) const;

    int ndim_ = 4;
    int dim_x_ = 8;
    bool is_obb_ = false;
    Matrix motion_mat_;
    Matrix update_mat_;
    double std_weight_position_ = 1.0 / 20.0;
    double std_weight_velocity_ = 1.0 / 160.0;
};

}  // namespace botsort

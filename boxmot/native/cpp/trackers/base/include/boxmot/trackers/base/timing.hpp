#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace boxmot::trackers::base {

// Timed steps are elapsed seconds divided by the fixed reference interval.
// Native velocity coordinates remain per reference frame; this is equivalent
// to converting velocity priors and process noise to seconds in Python.
// Zero explicitly selects the original discrete model.
inline void ValidateElapsedStep(const double dt) {
    if (!std::isfinite(dt) || dt < 0.0) {
        throw std::invalid_argument("Native dt must be finite and non-negative (zero selects fixed steps).");
    }
    if (!std::isfinite(dt * dt * dt)) {
        throw std::invalid_argument("dt produces an unrepresentable motion covariance.");
    }
}

inline std::pair<Eigen::MatrixXd, Eigen::MatrixXd> ElapsedMotion(
    const Eigen::MatrixXd& motion,
    const Eigen::MatrixXd& noise,
    const double dt
) {
    ValidateElapsedStep(dt);
    if (dt == 0.0) {
        return {motion, noise};
    }
    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(motion.rows(), motion.cols());
    const Eigen::MatrixXd generator = motion - identity;
    const Eigen::MatrixXd left = generator * noise;
    Eigen::MatrixXd transition = identity + dt * generator;
    Eigen::MatrixXd integrated = dt * noise + (dt * dt / 2.0) * (left + left.transpose()) +
        (dt * dt * dt / 3.0) * (left * generator.transpose());
    if (!transition.allFinite() || !integrated.allFinite()) {
        throw std::invalid_argument("dt produces an unrepresentable motion covariance.");
    }
    return {transition, integrated};
}

}  // namespace boxmot::trackers::base

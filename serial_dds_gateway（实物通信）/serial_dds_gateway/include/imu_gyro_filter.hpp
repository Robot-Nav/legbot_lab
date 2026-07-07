#pragma once

#include <array>
#include <chrono>
#include <cmath>
#include <iostream>

namespace serial_dds_gateway {

// Primary gyro bias calibration for sim2real (scheme A: gateway owns bias).
class ImuGyroFilter {
 public:
  ImuGyroFilter() { Reset(); }

  void Configure(bool enabled, double calib_seconds, double deadzone_rad, int min_samples = 50) {
    enabled_ = enabled;
    calib_seconds_ = calib_seconds;
    deadzone_rad_ = deadzone_rad;
    min_samples_ = min_samples;
    Reset();
  }

  void Reset() {
    start_ = std::chrono::steady_clock::now();
    sample_count_ = 0;
    sum_ = {0.0, 0.0, 0.0};
    bias_ = {0.0F, 0.0F, 0.0F};
    calib_done_ = false;
    logged_done_ = false;
  }

  bool calibration_done() const { return !enabled_ || calib_done_; }

  std::array<float, 3> bias() const { return bias_; }

  std::array<float, 3> Apply(float gx, float gy, float gz) {
    if (!enabled_) {
      return {gx, gy, gz};
    }

    if (!calib_done_) {
      sum_[0] += static_cast<double>(gx);
      sum_[1] += static_cast<double>(gy);
      sum_[2] += static_cast<double>(gz);
      ++sample_count_;

      const double elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
      if ((calib_seconds_ <= 0.0 || elapsed >= calib_seconds_) && sample_count_ >= min_samples_) {
        const double n = static_cast<double>(sample_count_);
        bias_[0] = static_cast<float>(sum_[0] / n);
        bias_[1] = static_cast<float>(sum_[1] / n);
        bias_[2] = static_cast<float>(sum_[2] / n);
        calib_done_ = true;
        if (!logged_done_) {
          logged_done_ = true;
          std::cout << "[INFO] IMU gyro bias calibrated over " << elapsed << "s (" << sample_count_
                    << " samples): gx=" << bias_[0] << " gy=" << bias_[1] << " gz=" << bias_[2]
                    << " rad/s\n";
        }
      }
    }

    const float bx = calib_done_ ? bias_[0]
                                 : static_cast<float>(sum_[0] / std::max(1, sample_count_));
    const float by = calib_done_ ? bias_[1]
                                 : static_cast<float>(sum_[1] / std::max(1, sample_count_));
    const float bz = calib_done_ ? bias_[2]
                                 : static_cast<float>(sum_[2] / std::max(1, sample_count_));

    return {Deadzone(gx - bx), Deadzone(gy - by), Deadzone(gz - bz)};
  }

 private:
  float Deadzone(float v) const {
    return std::fabs(v) < static_cast<float>(deadzone_rad_) ? 0.0F : v;
  }

  bool enabled_{true};
  double calib_seconds_{2.0};
  double deadzone_rad_{0.02};
  int min_samples_{50};

  std::chrono::steady_clock::time_point start_{};
  int sample_count_{0};
  std::array<double, 3> sum_{};
  std::array<float, 3> bias_{};
  bool calib_done_{false};
  bool logged_done_{false};
};

}  // namespace serial_dds_gateway

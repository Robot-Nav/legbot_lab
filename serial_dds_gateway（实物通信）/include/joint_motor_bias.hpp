#pragma once

#include "motor_map.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace serial_dds_gateway {

// Calf: motor 2 rev = joint 1 rev (2:1). Hip/thigh: 1:1.
inline constexpr float kCalfMotorPerJoint = 2.0f;

inline bool IsCalfJointIndex(size_t i) { return (i % 3) == 2; }

inline float MotorToJointScale(size_t i) {
  return IsCalfJointIndex(i) ? (1.0f / kCalfMotorPerJoint) : 1.0f;
}

inline float JointToMotorScale(size_t i) {
  return IsCalfJointIndex(i) ? kCalfMotorPerJoint : 1.0f;
}

// kp/kd/tau: 1:1 model -> motor. Position uses gear; impedance does not (RS02 PD in model units).
inline float ImpedanceToMotorScale(size_t /*i*/) { return 1.0f; }

inline float TorqueToMotorScale(size_t /*i*/) { return 1.0f; }

inline float TorqueToJointScale(size_t /*i*/) { return 1.0f; }

// Encoder sign vs URDF/model (+1 FR/RR, -1 FL/RL thigh & calf on this hardware).
inline constexpr std::array<float, 12> kDefaultMotorSign = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f,
};

inline float MotorSign(size_t i) { return kDefaultMotorSign[i]; }

inline float MotorSignTimesScale(size_t i) { return MotorSign(i) * MotorToJointScale(i); }

// Motor order FR, FL, RR, RL (hip/thigh/calf). Model/sim prone pose (MuJoCo LieDown).
inline constexpr std::array<float, 12> kDefaultProneModelQ = {
    -0.02f, 1.08f, -2.64f, 0.03f, 1.08f, -2.64f, -0.05f, 1.08f, -2.64f, 0.06f, 1.08f, -2.64f,
};

// Bias: bias[i]=sign[i]*gear[i]*q_motor_prone[i]-q_model_prone[i] (calf gear=1/2)
inline constexpr std::array<float, 12> kFatuProneBiasFromLog = {
    0.1598f, 0.1718f, 1.4753f, -0.1748f, 0.0919f, 1.4770f,
    -0.0706f, 0.1163f, 1.4254f, 0.0725f, 0.0672f, 1.4564f,
};

inline std::array<float, 12> FatuProneBiasFromLog() {
  return kFatuProneBiasFromLog;
}

// Verify mapping against log: sign*scale*q_motor - bias ≈ q_model at calib pose.
inline float MotorToJointFromRaw(size_t i, float q_motor) {
  return MotorSignTimesScale(i) * q_motor;
}

inline float JointToModelWithBias(size_t i, float q_motor, const std::array<float, 12>& bias) {
  return MotorToJointFromRaw(i, q_motor) - bias[i];
}

inline float ModelToMotorFromBias(size_t i, float q_model, const std::array<float, 12>& bias) {
  return (q_model + bias[i]) / MotorSignTimesScale(i);
}

inline std::array<float, 12> ParseJointBiasReference(const std::string& text) {
  std::array<float, 12> out{};
  std::stringstream ss(text);
  std::string token;
  size_t idx = 0;
  while (std::getline(ss, token, ',') && idx < out.size()) {
    if (token.empty()) continue;
    out[idx++] = std::stof(token);
  }
  if (idx != out.size()) {
    throw std::runtime_error("joint bias reference requires exactly 12 comma-separated values");
  }
  return out;
}

inline std::string ResolveGatewayConfigPath(const std::string& path) {
  if (path.empty()) return path;
  if (std::filesystem::exists(path)) return path;
  const std::string from_parent = (std::filesystem::path("..") / path).string();
  if (std::filesystem::exists(from_parent)) return from_parent;
  return path;
}

inline std::array<float, 12> LoadJointBiasFile(const std::string& path) {
  const std::string resolved = ResolveGatewayConfigPath(path);
  std::ifstream in(resolved);
  if (!in) throw std::runtime_error("cannot open joint bias file: " + path + " (tried " + resolved + ")");
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    return ParseJointBiasReference(line);
  }
  throw std::runtime_error("joint bias file is empty: " + resolved);
}

inline std::array<float, 12> LoadJointBiasReferenceFile(const std::string& path) {
  const std::string resolved = ResolveGatewayConfigPath(path);
  std::ifstream in(resolved);
  if (!in) throw std::runtime_error("cannot open joint bias reference file: " + path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    return ParseJointBiasReference(line);
  }
  throw std::runtime_error("joint bias reference file is empty: " + resolved);
}

// q_model = sign * gear_scale * q_motor - bias[i]; calf gear_scale=1/2; FL/RL thigh/calf sign=-1.
class JointMotorBiasMap {
 public:
  // apply_bias: use sign+bias mapping; run_online_calib: accumulate prone samples at startup.
  void Configure(bool apply_bias, bool run_online_calib, std::array<float, 12> model_reference_q) {
    apply_bias_ = apply_bias;
    run_online_calib_ = run_online_calib;
    model_reference_q_ = model_reference_q;
    bias_ = {};
    bias_calibrated_ = !run_online_calib;
    sample_count_ = 0;
    sum_q_ = {};
    calib_start_ = std::chrono::steady_clock::now();
  }

  void LoadBias(const std::array<float, 12>& bias, const char* source = "file") {
    bias_ = bias;
    bias_calibrated_ = true;
    std::cout << "[INFO] joint motor bias loaded from " << source << "\n";
    PrintBias();
  }

  bool apply_bias() const { return apply_bias_; }
  bool calibrated() const { return !run_online_calib_ || bias_calibrated_; }
  int sample_count() const { return sample_count_; }

  static int CountSeenMotors(const std::array<bool, 12>& seen) {
    int n = 0;
    for (bool s : seen) {
      if (s) ++n;
    }
    return n;
  }

  void MaybeLogCalibProgress(const std::array<bool, 12>& seen) {
    if (!run_online_calib_ || bias_calibrated_) return;
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - calib_start_).count();
    if (elapsed - last_progress_log_s_ < 1.0) return;
    last_progress_log_s_ = elapsed;
    const int seen_n = CountSeenMotors(seen);
    std::cout << "[WARN] joint bias calib waiting: " << seen_n << "/12 motors, " << sample_count_
              << " samples, " << elapsed << "s elapsed"
              << " (start fatu_ctrl Passive if motors not reporting)\n";
    if (seen_n < 12) {
      std::cout << "[WARN] missing motors:";
      for (size_t i = 0; i < seen.size(); ++i) {
        if (!seen[i]) std::cout << ' ' << kJointOrder[i];
      }
      std::cout << '\n';
    }
  }

  void AccumulateSample(const std::array<float, 12>& q_motor, const std::array<bool, 12>& seen) {
    if (!run_online_calib_ || bias_calibrated_) return;
    for (size_t i = 0; i < q_motor.size(); ++i) {
      if (!seen[i]) return;
    }
    for (size_t i = 0; i < q_motor.size(); ++i) {
      sum_q_[i] += static_cast<double>(q_motor[i]);
    }
    ++sample_count_;
  }

  bool TryFinishCalibration(double calib_seconds, int min_samples, double timeout_seconds) {
    if (!run_online_calib_ || bias_calibrated_) return true;
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - calib_start_).count();
    if (timeout_seconds > 0.0 && elapsed >= timeout_seconds && sample_count_ < min_samples) {
      std::cerr << "[WARN] joint bias calib timeout after " << elapsed
                << "s with only " << sample_count_ << " samples; using zero bias\n";
      bias_ = {};
      bias_calibrated_ = true;
      return true;
    }
    if (sample_count_ < min_samples) return false;
    if (calib_seconds > 0.0 && elapsed < calib_seconds) return false;

    const double n = static_cast<double>(sample_count_);
    std::cout << "[INFO] joint motor bias calibration over " << elapsed << "s (" << sample_count_
              << " samples, robot prone/still)\n";
    std::cout << "[INFO] motor raw q: [";
    for (size_t i = 0; i < bias_.size(); ++i) {
      if (i > 0) std::cout << ", ";
      const float q_motor = static_cast<float>(sum_q_[i] / n);
      std::cout << std::fixed << std::setprecision(4) << q_motor;
    }
    std::cout << "]\n";
    std::cout << "[INFO] joint q after sign+gear (calf /2): [";
    for (size_t i = 0; i < bias_.size(); ++i) {
      if (i > 0) std::cout << ", ";
      const float q_motor = static_cast<float>(sum_q_[i] / n);
      const float q_joint = MotorSignTimesScale(i) * q_motor;
      std::cout << std::fixed << std::setprecision(4) << q_joint;
      bias_[i] = q_joint - model_reference_q_[i];
    }
    std::cout << "]\n";
    std::cout << "[INFO] model reference q: [";
    for (size_t i = 0; i < model_reference_q_.size(); ++i) {
      if (i > 0) std::cout << ", ";
      std::cout << std::fixed << std::setprecision(4) << model_reference_q_[i];
    }
    std::cout << "]\n";
    bias_calibrated_ = true;
    PrintBias();
    return true;
  }

  float MotorToModel(size_t i, float q_motor) const {
    const float q_joint = MotorSignTimesScale(i) * q_motor;
    if (!apply_bias_ || !bias_calibrated_) return q_joint;
    return q_joint - bias_[i];
  }

  float ModelToMotor(size_t i, float q_model) const {
    const float q_joint = (apply_bias_ && bias_calibrated_) ? (q_model + bias_[i]) : q_model;
    return q_joint / MotorSignTimesScale(i);
  }

  float MotorToModelDq(size_t i, float dq_motor) const { return MotorSignTimesScale(i) * dq_motor; }

  float ModelToMotorDq(size_t i, float dq_model) const { return dq_model / MotorSignTimesScale(i); }

  float ModelToMotorKp(size_t i, float kp_model) const { return ImpedanceToMotorScale(i) * kp_model; }

  float ModelToMotorKd(size_t i, float kd_model) const { return ImpedanceToMotorScale(i) * kd_model; }

  float ModelToMotorTau(size_t i, float tau_model) const { return TorqueToMotorScale(i) * tau_model; }

  float MotorToModelTau(size_t i, float tau_motor) const { return TorqueToJointScale(i) * tau_motor; }

  const std::array<float, 12>& bias() const { return bias_; }

 private:
  void PrintBias() const {
    std::cout << "[INFO] joint bias (joint-model space): [";
    for (size_t i = 0; i < bias_.size(); ++i) {
      if (i > 0) std::cout << ", ";
      std::cout << std::fixed << std::setprecision(4) << bias_[i];
    }
    std::cout << "]\n";
    std::cout << "[INFO] mapping: q=sign*gear*q_motor-bias (calf gear 1/2); kp/kd/tau 1:1; FL/RL thigh/calf sign=-1\n";
  }

  bool apply_bias_{false};
  bool run_online_calib_{false};
  bool bias_calibrated_{true};
  int sample_count_{0};
  std::array<float, 12> model_reference_q_{};
  std::array<float, 12> bias_{};
  std::array<double, 12> sum_q_{};
  std::chrono::steady_clock::time_point calib_start_{};
  double last_progress_log_s_{-1.0};
};

}  // namespace serial_dds_gateway

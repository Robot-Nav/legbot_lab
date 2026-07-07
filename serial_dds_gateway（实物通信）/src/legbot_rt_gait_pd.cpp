// Real-time-ish Legbot gait executor: subscribe rt/lowstate, publish rt/lowcmd.
// This process never touches the serial bus; dds_to_serial_gateway owns motor IO.

#include "motor_map.hpp"

#include "unitree/robot/channel/channel_factory.hpp"
#include "unitree/dds_wrapper/robots/go2/go2_pub.h"
#include "unitree/dds_wrapper/robots/go2/go2_sub.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sstream>
#include <thread>
#include <vector>

namespace {

using LowCmdPublisher = unitree::robot::go2::publisher::LowCmd;
using LowStateSubscriber = unitree::robot::go2::subscription::LowState;
using LowCmdMsg = unitree_go::msg::dds_::LowCmd_;
using LowStateMsg = unitree_go::msg::dds_::LowState_;
using Clock = std::chrono::steady_clock;

std::atomic<bool> g_running{true};

constexpr const char* kLowCmdTopic = "rt/lowcmd";
constexpr const char* kLowStateTopic = "rt/lowstate";
constexpr double kPi = 3.14159265358979323846;
constexpr size_t kNumJoints = 12;
constexpr size_t kNumLegs = 4;

constexpr double kAbadLinkLength = 0.0975;
constexpr double kHipLinkLength = 0.1985;
constexpr double kKneeLinkLength = 0.214;

enum class Leg : size_t { FL = 0, FR = 1, RL = 2, RR = 3 };

struct Vec3 {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

struct LegInfo {
  Leg leg;
  const char* name;
  size_t dds_base;
  double side_sign;
  Vec3 abad_offset_body;
};

constexpr std::array<LegInfo, kNumLegs> kLegs = {{
    {Leg::FL, "FL", 3, 1.0, {0.18453, 0.051, 0.0}},
    {Leg::FR, "FR", 0, -1.0, {0.18453, -0.051, 0.0}},
    {Leg::RL, "RL", 9, 1.0, {-0.18453, 0.051, 0.0}},
    {Leg::RR, "RR", 6, -1.0, {-0.18453, -0.051, 0.0}},
}};

// Stand pose in DDS joint order:
//   FR_hip, FR_thigh, FR_calf,
//   FL_hip, FL_thigh, FL_calf,
//   RR_hip, RR_thigh, RR_calf,
//   RL_hip, RL_thigh, RL_calf.
// Model stand pose requested for real-machine test.
constexpr std::array<double, kNumJoints> kStandQ = {
    -0.0, 0.9, -1.8,
     0.0, 0.9, -1.8,
    -0.0, 0.9, -1.8,
     0.0, 0.9, -1.8,
};

// Down/prone pose in DDS joint order:
//   FR_hip, FR_thigh, FR_calf,
//   FL_hip, FL_thigh, FL_calf,
//   RR_hip, RR_thigh, RR_calf,
//   RL_hip, RL_thigh, RL_calf.
// This matches the measured safe crouch/down feedback on the current LEGBOT.
// Calf is kept at the configured soft-limit boundary.
constexpr std::array<double, kNumJoints> kDownQ = {
    -0.02, 1.08, -2.6387,
     0.03, 1.08, -2.6387,
    -0.05, 1.08, -2.6387,
     0.06, 1.08, -2.6387,
};

struct JointSoftLimit {
  double lo;
  double hi;
};

struct Args {
  std::string network = "lo";
  std::string mode = "trot";
  double duration_s = 8.0;
  double cmd_hz = 100.0;
  double gait_frequency_hz = 0.5;
  double gait_duty = 0.65;
  double swing_height = 0.05;
  double step_length = 0.0;
  double kp = 20.0;
  double kd = 3.0;
  double swing_hip_q_step_limit = 0.010;
  double swing_thigh_q_step_limit = 0.014;
  double swing_calf_q_step_limit = 0.020;
  double startup_ramp_seconds = 12.0;
  double prehold_seconds = 1.0;
  double wait_lowstate_s = 5.0;
  double lowstate_timeout_s = 0.25;
  double max_q_error = 0.45;
  double max_tilt_rad = 0.08;
  bool robot_standing_supported = false;
  bool i_accept_risk = false;
  bool disable_on_exit = false;
  bool return_down_on_exit = false;
  double return_ramp_seconds = 5.0;
  double exit_disable_seconds = 1.0;
  bool dry_run = false;
  bool feedback_bootstrap = false;
  bool allow_long_duration = false;
  bool help = false;
};

struct LoopStats {
  uint64_t loop_count{0};
  uint64_t publish_count{0};
  uint64_t lowstate_count{0};
  double lowstate_interval_sum_s{0.0};
  double lowstate_interval_min_s{std::numeric_limits<double>::infinity()};
  double lowstate_interval_max_s{0.0};
  double loop_ms_sum{0.0};
  double loop_ms_max{0.0};
  double max_qerr{0.0};
  double max_tilt{0.0};
  std::vector<double> publish_times_s;
};

void OnSignal(int) { g_running = false; }

double SecondsSince(const Clock::time_point& t0, const Clock::time_point& t) {
  return std::chrono::duration<double>(t - t0).count();
}

double Clamp(double v, double lo, double hi) {
  return std::min(std::max(v, lo), hi);
}

double SmoothStep(double x) {
  x = Clamp(x, 0.0, 1.0);
  return x * x * (3.0 - 2.0 * x);
}

double MinJerk(double x) {
  x = Clamp(x, 0.0, 1.0);
  return 10.0 * x * x * x - 15.0 * x * x * x * x + 6.0 * x * x * x * x * x;
}

bool IsFinite(double v) { return std::isfinite(v); }

bool IsFrontLeg(const LegInfo& leg) {
  return leg.leg == Leg::FL || leg.leg == Leg::FR;
}

JointSoftLimit SoftLimitForJoint(const LegInfo& leg, size_t leg_joint_index) {
  if (leg_joint_index == 0) return {-0.73304, 0.73304};
  if (leg_joint_index == 1) {
    return IsFrontLeg(leg) ? JointSoftLimit{-1.559, 3.1298} : JointSoftLimit{-0.51181, 4.177};
  }
  return {-2.6387, -0.7854};
}

double StepLimitForJoint(size_t joint_index, const Args& args) {
  switch (joint_index % 3) {
    case 0:
      return args.swing_hip_q_step_limit;
    case 1:
      return args.swing_thigh_q_step_limit;
    default:
      return args.swing_calf_q_step_limit;
  }
}

const LegInfo& LegFromIndex(size_t idx) { return kLegs.at(idx); }

const LegInfo& LegFromJointIndex(size_t joint_index) {
  for (const auto& leg : kLegs) {
    if (joint_index >= leg.dds_base && joint_index < leg.dds_base + 3) return leg;
  }
  throw std::runtime_error("joint index is not part of a Legbot leg");
}

Vec3 Add(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 Sub(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }

Vec3 CalcLegFkBody(const LegInfo& leg, const std::array<double, 3>& q) {
  const double l1 = leg.side_sign * kAbadLinkLength;
  const double l2 = -kHipLinkLength;
  const double l3 = -kKneeLinkLength;

  const double s1 = std::sin(q[0]);
  const double s2 = std::sin(q[1]);
  const double s3 = std::sin(q[2]);
  const double c1 = std::cos(q[0]);
  const double c2 = std::cos(q[1]);
  const double c3 = std::cos(q[2]);
  const double c23 = c2 * c3 - s2 * s3;
  const double s23 = s2 * c3 + c2 * s3;

  const Vec3 hip_frame{
      l3 * s23 + l2 * s2,
      -l3 * s1 * c23 + l1 * c1 - l2 * c2 * s1,
      l3 * c1 * c23 + l1 * s1 + l2 * c1 * c2,
  };
  return Add(leg.abad_offset_body, hip_frame);
}

bool CalcLegIkBody(const LegInfo& leg, const Vec3& foot_pos_body, std::array<double, 3>* q_out) {
  const Vec3 p = Sub(foot_pos_body, leg.abad_offset_body);
  const double px = p.x;
  const double py = p.y;
  const double pz = p.z;
  const double b2y = kAbadLinkLength * leg.side_sign;
  const double b3z = -kHipLinkLength;
  const double b4z = -kKneeLinkLength;
  const double a = kAbadLinkLength;

  const double c = std::sqrt(px * px + py * py + pz * pz);
  const double b_sq = c * c - a * a;
  if (!IsFinite(c) || b_sq < -1.0e-8) return false;
  const double b = std::sqrt(std::max(b_sq, 1.0e-12));

  const double lateral_sq = py * py + pz * pz - b2y * b2y;
  if (lateral_sq < -1.0e-8) return false;
  const double lateral = std::sqrt(std::max(lateral_sq, 1.0e-12));
  const double q1 = std::atan2(pz * b2y + py * lateral, py * b2y - pz * lateral);

  const double denom = 2.0 * std::abs(b3z * b4z);
  if (denom < 1.0e-12) return false;
  const double temp_raw = (b3z * b3z + b4z * b4z - b * b) / denom;
  if (temp_raw < -1.0001 || temp_raw > 1.0001) return false;
  const double temp = Clamp(temp_raw, -1.0, 1.0);
  const double q3 = -(kPi - std::acos(temp));

  const double a1 = py * std::sin(q1) - pz * std::cos(q1);
  const double a2 = px;
  const double m1 = b4z * std::sin(q3);
  const double m2 = b3z + b4z * std::cos(q3);
  const double q2 = std::atan2(m1 * a1 + m2 * a2, m1 * a2 - m2 * a1);

  if (!IsFinite(q1) || !IsFinite(q2) || !IsFinite(q3)) return false;
  *q_out = {q1, q2, q3};
  return true;
}

std::array<Vec3, kNumLegs> StandFeetBody() {
  std::array<Vec3, kNumLegs> feet{};
  for (size_t i = 0; i < kLegs.size(); ++i) {
    const auto& leg = kLegs[i];
    const std::array<double, 3> q{
        kStandQ[leg.dds_base + 0],
        kStandQ[leg.dds_base + 1],
        kStandQ[leg.dds_base + 2],
    };
    feet[i] = CalcLegFkBody(leg, q);
  }
  return feet;
}

double OpenLoopLegPhase(const LegInfo& leg, double gait_time_s, const Args& args) {
  const double period = 1.0 / std::max(args.gait_frequency_hz, 1.0e-9);
  const double offset = (leg.leg == Leg::FL || leg.leg == Leg::RR) ? 0.0 : 0.5;
  double phase = std::fmod(gait_time_s / period + offset, 1.0);
  if (phase < 0.0) phase += 1.0;
  return phase;
}

bool IsSwingLeg(const LegInfo& leg, double gait_time_s, const Args& args) {
  const double swing_frac = std::max(1.0 - args.gait_duty, 1.0e-6);
  return OpenLoopLegPhase(leg, gait_time_s, args) < swing_frac;
}

double SwingAlpha(const LegInfo& leg, double gait_time_s, const Args& args) {
  const double swing_frac = std::max(1.0 - args.gait_duty, 1.0e-6);
  return Clamp(OpenLoopLegPhase(leg, gait_time_s, args) / swing_frac, 0.0, 1.0);
}

double StanceAlpha(const LegInfo& leg, double gait_time_s, const Args& args) {
  const double phase = OpenLoopLegPhase(leg, gait_time_s, args);
  const double swing_frac = std::max(1.0 - args.gait_duty, 1.0e-6);
  const double stance_frac = std::max(args.gait_duty, 1.0e-6);
  if (phase < swing_frac) return 0.0;
  return Clamp((phase - swing_frac) / stance_frac, 0.0, 1.0);
}

std::array<int, kNumLegs> ContactMask(double gait_time_s, const Args& args) {
  std::array<int, kNumLegs> mask{};
  for (size_t i = 0; i < kLegs.size(); ++i) {
    mask[i] = IsSwingLeg(kLegs[i], gait_time_s, args) ? 0 : 1;
  }
  return mask;
}

std::string MaskText(const std::array<int, kNumLegs>& mask) {
  std::string out;
  out.reserve(mask.size());
  for (const int v : mask) out += (v ? '1' : '0');
  return out;
}

void ApplySoftLimits(std::array<double, kNumJoints>* q) {
  for (const auto& leg : kLegs) {
    for (size_t j = 0; j < 3; ++j) {
      const auto lim = SoftLimitForJoint(leg, j);
      (*q)[leg.dds_base + j] = Clamp((*q)[leg.dds_base + j], lim.lo, lim.hi);
    }
  }
}

std::array<double, kNumJoints> RateLimitTargets(const std::array<double, kNumJoints>& q_raw,
                                                std::array<double, kNumJoints> q_prev,
                                                const std::array<double, kNumJoints>& q_now,
                                                const std::array<int, kNumLegs>& contact_mask,
                                                const std::array<int, kNumLegs>& prev_contact_mask,
                                                const Args& args) {
  for (size_t leg_idx = 0; leg_idx < kLegs.size(); ++leg_idx) {
    if (prev_contact_mask[leg_idx] == 1 && contact_mask[leg_idx] == 0) {
      const auto& leg = kLegs[leg_idx];
      for (size_t j = 0; j < 3; ++j) q_prev[leg.dds_base + j] = q_now[leg.dds_base + j];
    }
  }

  std::array<double, kNumJoints> out{};
  for (size_t i = 0; i < out.size(); ++i) {
    const double step = StepLimitForJoint(i, args);
    const double delta = q_raw[i] - q_prev[i];
    out[i] = q_prev[i] + Clamp(delta, -step, step);
  }
  return out;
}

std::array<double, kNumJoints> ComputeRawTarget(const Args& args, double gait_time_s,
                                                const std::array<Vec3, kNumLegs>& stand_feet,
                                                const std::array<double, kNumJoints>& q_prev_target,
                                                std::array<int, kNumLegs>* contact_mask) {
  std::array<double, kNumJoints> q_raw = kStandQ;
  if (args.mode == "stand") {
    contact_mask->fill(1);
    return q_raw;
  }

  *contact_mask = ContactMask(gait_time_s, args);
  for (size_t leg_idx = 0; leg_idx < kLegs.size(); ++leg_idx) {
    const auto& leg = LegFromIndex(leg_idx);
    Vec3 foot = stand_feet[leg_idx];
    if ((*contact_mask)[leg_idx] == 0) {
      const double s = MinJerk(SwingAlpha(leg, gait_time_s, args));
      foot.x += (-0.5 + s) * args.step_length;
      foot.z += args.swing_height * 4.0 * s * (1.0 - s);
    } else {
      const double alpha = SmoothStep(StanceAlpha(leg, gait_time_s, args));
      foot.x += (0.5 - alpha) * args.step_length;
    }

    std::array<double, 3> q_leg{};
    if (!CalcLegIkBody(leg, foot, &q_leg)) {
      for (size_t j = 0; j < 3; ++j) q_raw[leg.dds_base + j] = q_prev_target[leg.dds_base + j];
      continue;
    }
    for (size_t j = 0; j < 3; ++j) q_raw[leg.dds_base + j] = q_leg[j];
  }
  ApplySoftLimits(&q_raw);
  return q_raw;
}

void PrintHelp(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n\n"
      << "DDS-only Legbot realtime IK+PD gait executor. Publishes " << kLowCmdTopic
      << " and subscribes " << kLowStateTopic << ".\n\n"
      << "Required safety gates for non-dry-run control:\n"
      << "  --robot-standing-supported\n"
      << "  --i-accept-risk\n"
      << "Dry-run does not create a LowCmd publisher and does not require these gates.\n"
      << "Feedback-bootstrap publishes zero kp/kd/tau LowCmd only and does not require these gates.\n\n"
      << "Main options:\n"
      << "  --network IFACE                       DDS network interface (default: lo)\n"
      << "  --mode stand|trot                     Control mode (default: trot)\n"
      << "  --dry-run                             Subscribe LowState only; never publish LowCmd\n"
      << "  --feedback-bootstrap                  Publish mode=1, kp=kd=tau=0 to wake motor feedback only\n"
      << "  --duration S                          Control duration after ramp/prehold, or dry-run/bootstrap duration (default: 8)\n"
      << "  --cmd-hz HZ                           LowCmd publish rate (default: 100)\n"
      << "  --gait-frequency-hz HZ                Trot base gait frequency (default: 0.5)\n"
      << "  --gait-duty D                         Stance duty factor, 0<D<1 (default: 0.65)\n"
      << "  --swing-height M                      Swing foot z bump (default: 0.05)\n"
      << "  --step-length M                       Body-frame stance sweep length (default: 0.0)\n"
      << "  --kp GAIN                             Position gain for all 12 motors (default: 20)\n"
      << "  --kd GAIN                             Damping gain for all 12 motors (default: 3)\n"
      << "  --swing-hip-q-step-limit RAD          Per tick q target limit (default: 0.010)\n"
      << "  --swing-thigh-q-step-limit RAD        Per tick q target limit (default: 0.014)\n"
      << "  --swing-calf-q-step-limit RAD         Per tick q target limit (default: 0.020)\n"
      << "  --startup-ramp-seconds S              Current q -> stand ramp (default: 12)\n"
      << "  --prehold-seconds S                   Stand hold before control (default: 1)\n"
      << "  --wait-lowstate-s S                   Initial LowState wait timeout (default: 5)\n"
      << "  --lowstate-timeout-s S                Runtime LowState timeout (default: 0.25)\n"
      << "  --max-q-error RAD                     Abort on max |q-qtarget| (default: 0.45)\n"
      << "  --max-tilt-rad RAD                    Abort on max |roll|/|pitch| (default: 0.08)\n"
      << "  --allow-long-duration                 Allow control --duration >12s or bootstrap >5s\n"
      << "  --disable-on-exit                     Publish mode=0 disable on normal exit too\n"
      << "  --return-down-on-exit                 On normal completion, ramp current q -> down pose, then disable\n"
      << "  --return-ramp-seconds S               Duration for return-down ramp (default: 5)\n"
      << "  --exit-disable-seconds S              Duration for final mode=0 disable publish (default: 1)\n"
      << "  --help                                Show this help\n";
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto need = [&](const char* name) {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return std::string(argv[++i]);
    };

    if (k == "--help" || k == "-h") args.help = true;
    else if (k == "--network") args.network = need("--network");
    else if (k == "--mode") args.mode = need("--mode");
    else if (k == "--dry-run") args.dry_run = true;
    else if (k == "--feedback-bootstrap") args.feedback_bootstrap = true;
    else if (k == "--duration") args.duration_s = std::stod(need("--duration"));
    else if (k == "--cmd-hz") args.cmd_hz = std::stod(need("--cmd-hz"));
    else if (k == "--gait-frequency-hz") args.gait_frequency_hz = std::stod(need("--gait-frequency-hz"));
    else if (k == "--gait-duty") args.gait_duty = std::stod(need("--gait-duty"));
    else if (k == "--swing-height") args.swing_height = std::stod(need("--swing-height"));
    else if (k == "--step-length") args.step_length = std::stod(need("--step-length"));
    else if (k == "--kp") args.kp = std::stod(need("--kp"));
    else if (k == "--kd") args.kd = std::stod(need("--kd"));
    else if (k == "--swing-hip-q-step-limit") args.swing_hip_q_step_limit = std::stod(need("--swing-hip-q-step-limit"));
    else if (k == "--swing-thigh-q-step-limit") args.swing_thigh_q_step_limit = std::stod(need("--swing-thigh-q-step-limit"));
    else if (k == "--swing-calf-q-step-limit") args.swing_calf_q_step_limit = std::stod(need("--swing-calf-q-step-limit"));
    else if (k == "--startup-ramp-seconds") args.startup_ramp_seconds = std::stod(need("--startup-ramp-seconds"));
    else if (k == "--prehold-seconds") args.prehold_seconds = std::stod(need("--prehold-seconds"));
    else if (k == "--wait-lowstate-s") args.wait_lowstate_s = std::stod(need("--wait-lowstate-s"));
    else if (k == "--lowstate-timeout-s") args.lowstate_timeout_s = std::stod(need("--lowstate-timeout-s"));
    else if (k == "--max-q-error") args.max_q_error = std::stod(need("--max-q-error"));
    else if (k == "--max-tilt-rad") args.max_tilt_rad = std::stod(need("--max-tilt-rad"));
    else if (k == "--robot-standing-supported") args.robot_standing_supported = true;
    else if (k == "--i-accept-risk") args.i_accept_risk = true;
    else if (k == "--allow-long-duration") args.allow_long_duration = true;
    else if (k == "--disable-on-exit") args.disable_on_exit = true;
    else if (k == "--return-down-on-exit") args.return_down_on_exit = true;
    else if (k == "--return-ramp-seconds") args.return_ramp_seconds = std::stod(need("--return-ramp-seconds"));
    else if (k == "--exit-disable-seconds") args.exit_disable_seconds = std::stod(need("--exit-disable-seconds"));
    else throw std::runtime_error("unknown option: " + k);
  }
  return args;
}

void ValidateArgs(const Args& args) {
  if (args.help) return;
  if (args.dry_run && args.feedback_bootstrap) {
    throw std::runtime_error("--dry-run and --feedback-bootstrap are mutually exclusive");
  }
  if (!args.dry_run && !args.feedback_bootstrap) {
    if (!args.robot_standing_supported) {
      throw std::runtime_error("refusing to start without --robot-standing-supported");
    }
    if (!args.i_accept_risk) {
      throw std::runtime_error("refusing to start without --i-accept-risk");
    }
    if (args.duration_s > 12.0 && !args.allow_long_duration) {
      throw std::runtime_error("refusing non-dry-run --duration > 12 without --allow-long-duration");
    }
  }
  if (args.feedback_bootstrap && args.duration_s > 5.0 && !args.allow_long_duration) {
    throw std::runtime_error("refusing --feedback-bootstrap --duration > 5 without --allow-long-duration");
  }
  if (args.mode != "stand" && args.mode != "trot") {
    throw std::runtime_error("--mode must be stand or trot");
  }
  if (args.duration_s < 0.0) throw std::runtime_error("--duration must be non-negative");
  if (args.cmd_hz <= 0.0) throw std::runtime_error("--cmd-hz must be positive");
  if (args.gait_frequency_hz <= 0.0) throw std::runtime_error("--gait-frequency-hz must be positive");
  if (!(args.gait_duty > 0.0 && args.gait_duty < 1.0)) throw std::runtime_error("--gait-duty must satisfy 0<D<1");
  if (args.swing_height < 0.0) throw std::runtime_error("--swing-height must be non-negative");
  if (args.kp < 0.0 || args.kd < 0.0) throw std::runtime_error("--kp/--kd must be non-negative");
  if (args.startup_ramp_seconds < 0.0 || args.prehold_seconds < 0.0) {
    throw std::runtime_error("--startup-ramp-seconds/--prehold-seconds must be non-negative");
  }
  if (args.return_ramp_seconds < 0.0 || args.exit_disable_seconds < 0.0) {
    throw std::runtime_error("--return-ramp-seconds/--exit-disable-seconds must be non-negative");
  }
  if (args.lowstate_timeout_s <= 0.0 || args.wait_lowstate_s <= 0.0) {
    throw std::runtime_error("--lowstate-timeout-s/--wait-lowstate-s must be positive");
  }
  if (args.max_q_error <= 0.0 || args.max_tilt_rad <= 0.0) {
    throw std::runtime_error("--max-q-error/--max-tilt-rad must be positive");
  }
}

LowStateMsg SnapshotLowState(const std::shared_ptr<LowStateSubscriber>& sub) {
  std::lock_guard<std::mutex> lock(sub->mutex_);
  return sub->msg_;
}

std::array<double, kNumJoints> ReadQ(const LowStateMsg& msg) {
  std::array<double, kNumJoints> q{};
  for (size_t i = 0; i < q.size(); ++i) {
    q[i] = static_cast<double>(msg.motor_state()[i].q());
  }
  return q;
}

std::array<double, kNumJoints> ReadDq(const LowStateMsg& msg) {
  std::array<double, kNumJoints> dq{};
  for (size_t i = 0; i < dq.size(); ++i) {
    dq[i] = static_cast<double>(msg.motor_state()[i].dq());
  }
  return dq;
}

std::array<float, 4> ReadImuQuatWxyz(const LowStateMsg& msg) {
  return msg.imu_state().quaternion();
}

std::array<double, 3> ReadGyro(const LowStateMsg& msg) {
  const auto gyro = msg.imu_state().gyroscope();
  return {static_cast<double>(gyro[0]), static_cast<double>(gyro[1]), static_cast<double>(gyro[2])};
}

template <typename ArrayT>
std::string ArrayText(const ArrayT& values, int precision) {
  std::ostringstream oss;
  oss << "[" << std::fixed << std::setprecision(precision);
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) oss << ", ";
    oss << static_cast<double>(values[i]);
  }
  oss << "]";
  return oss.str();
}

std::string LostFlagsText(const LowStateMsg& msg) {
  std::ostringstream oss;
  bool any = false;
  for (size_t i = 0; i < kNumJoints; ++i) {
    const auto lost = msg.motor_state()[i].lost();
    if (lost != 0u) {
      if (any) oss << ",";
      oss << serial_dds_gateway::kJointOrder[i] << ":" << static_cast<unsigned>(lost);
      any = true;
    }
  }
  return any ? oss.str() : std::string("none");
}

bool HasLostMotor(const LowStateMsg& msg, std::string* lost_joint) {
  for (size_t i = 0; i < kNumJoints; ++i) {
    if (msg.motor_state()[i].lost() != 0u) {
      if (lost_joint) *lost_joint = std::string(serial_dds_gateway::kJointOrder[i]);
      return true;
    }
  }
  return false;
}

std::array<double, 3> QuatWxyzToRpy(const std::array<float, 4>& quat) {
  double w = quat[0];
  double x = quat[1];
  double y = quat[2];
  double z = quat[3];
  const double norm = std::sqrt(w * w + x * x + y * y + z * z);
  if (norm <= 1.0e-9) return {0.0, 0.0, 0.0};
  w /= norm;
  x /= norm;
  y /= norm;
  z /= norm;

  const double sinr_cosp = 2.0 * (w * x + y * z);
  const double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
  const double roll = std::atan2(sinr_cosp, cosr_cosp);

  const double sinp = 2.0 * (w * y - z * x);
  const double pitch = std::abs(sinp) >= 1.0 ? std::copysign(kPi / 2.0, sinp) : std::asin(sinp);

  const double siny_cosp = 2.0 * (w * z + x * y);
  const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
  const double yaw = std::atan2(siny_cosp, cosy_cosp);
  return {roll, pitch, yaw};
}

double TiltFromLowState(const LowStateMsg& msg) {
  const auto rpy = QuatWxyzToRpy(msg.imu_state().quaternion());
  return std::max(std::abs(rpy[0]), std::abs(rpy[1]));
}

double MaxQError(const std::array<double, kNumJoints>& q_now, const std::array<double, kNumJoints>& q_target,
                 std::string* joint_name = nullptr) {
  double max_err = 0.0;
  size_t max_i = 0;
  for (size_t i = 0; i < kNumJoints; ++i) {
    const double err = std::abs(q_now[i] - q_target[i]);
    if (err > max_err) {
      max_err = err;
      max_i = i;
    }
  }
  if (joint_name) *joint_name = std::string(serial_dds_gateway::kJointOrder[max_i]);
  return max_err;
}

void FillLowCmd(LowCmdMsg& cmd, uint8_t mode, const std::array<double, kNumJoints>& q_target,
                double kp, double kd) {
  cmd.head() = {0xFE, 0xEF};
  cmd.level_flag() = 0xFF;
  cmd.gpio() = 0;
  for (size_t i = 0; i < cmd.motor_cmd().size(); ++i) {
    auto& m = cmd.motor_cmd()[i];
    if (i < kNumJoints && mode != 0) {
      m.mode(mode);
      m.q(static_cast<float>(q_target[i]));
      m.dq(0.0f);
      m.kp(static_cast<float>(kp));
      m.kd(static_cast<float>(kd));
      m.tau(0.0f);
    } else {
      m.mode(0);
      m.q(0.0f);
      m.dq(0.0f);
      m.kp(0.0f);
      m.kd(0.0f);
      m.tau(0.0f);
    }
  }
}

bool PublishLowCmd(LowCmdPublisher& pub, uint8_t mode, const std::array<double, kNumJoints>& q_target,
                   double kp, double kd, LoopStats* stats, const Clock::time_point& stats_t0) {
  const auto deadline = Clock::now() + std::chrono::milliseconds(50);
  while (Clock::now() < deadline) {
    if (pub.trylock()) {
      FillLowCmd(pub.msg_, mode, q_target, kp, kd);
      pub.unlockAndPublish();
      if (stats) {
        ++stats->publish_count;
        stats->publish_times_s.push_back(SecondsSince(stats_t0, Clock::now()));
      }
      return true;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }
  return false;
}

void PublishDisableFor(LowCmdPublisher* pub, const Args& args, LoopStats* stats,
                       const Clock::time_point& stats_t0, double seconds) {
  if (!pub || seconds <= 0.0) return;
  std::array<double, kNumJoints> zero_q{};
  const double hz = std::max(args.cmd_hz, 1.0);
  const int count = std::max(1, static_cast<int>(std::ceil(seconds * hz)));
  for (int i = 0; i < count; ++i) {
    PublishLowCmd(*pub, 0, zero_q, 0.0, 0.0, stats, stats_t0);
    std::this_thread::sleep_for(std::chrono::duration<double>(1.0 / hz));
  }
}

void PublishDisableBurst(LowCmdPublisher* pub, const Args& args, LoopStats* stats,
                         const Clock::time_point& stats_t0) {
  PublishDisableFor(pub, args, stats, stats_t0, args.exit_disable_seconds);
}

void CheckSafety(const LowStateMsg& msg, const std::array<double, kNumJoints>& q_target,
                 const Args& args, LoopStats* stats) {
  std::string lost_joint;
  if (HasLostMotor(msg, &lost_joint)) {
    throw std::runtime_error("motor_lost:" + lost_joint);
  }
  const auto q_now = ReadQ(msg);
  std::string qerr_joint;
  const double qerr = MaxQError(q_now, q_target, &qerr_joint);
  stats->max_qerr = std::max(stats->max_qerr, qerr);
  if (qerr > args.max_q_error) {
    throw std::runtime_error("max_q_error:" + qerr_joint);
  }
  const double tilt = TiltFromLowState(msg);
  stats->max_tilt = std::max(stats->max_tilt, tilt);
  if (tilt > args.max_tilt_rad) {
    throw std::runtime_error("max_tilt");
  }
}

void WaitForValidLowState(const std::shared_ptr<LowStateSubscriber>& sub, const Args& args) {
  const auto deadline = Clock::now() + std::chrono::duration<double>(args.wait_lowstate_s);
  while (g_running && Clock::now() < deadline) {
    if (!sub->isTimeout()) {
      const auto msg = SnapshotLowState(sub);
      if (!HasLostMotor(msg, nullptr)) return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (!g_running) throw std::runtime_error("interrupted while waiting for LowState");
  if (sub->isTimeout()) throw std::runtime_error("no rt/lowstate received");
  throw std::runtime_error("motor feedback is still lost");
}

void WaitForAnyLowState(const std::shared_ptr<LowStateSubscriber>& sub, const Args& args) {
  const auto deadline = Clock::now() + std::chrono::duration<double>(args.wait_lowstate_s);
  while (g_running && Clock::now() < deadline) {
    if (!sub->isTimeout()) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (!g_running) throw std::runtime_error("interrupted while waiting for LowState");
  throw std::runtime_error("no rt/lowstate received");
}

void RecordLowStateSample(const LowStateMsg& msg, LoopStats* stats, Clock::time_point* last_sample_t,
                          uint32_t* last_tick) {
  const uint32_t tick = msg.tick();
  if (stats->lowstate_count > 0 && tick == *last_tick) return;
  const auto now = Clock::now();
  if (stats->lowstate_count > 0) {
    const double dt = SecondsSince(*last_sample_t, now);
    stats->lowstate_interval_sum_s += dt;
    stats->lowstate_interval_min_s = std::min(stats->lowstate_interval_min_s, dt);
    stats->lowstate_interval_max_s = std::max(stats->lowstate_interval_max_s, dt);
  }
  *last_sample_t = now;
  *last_tick = tick;
  ++stats->lowstate_count;
  stats->max_tilt = std::max(stats->max_tilt, TiltFromLowState(msg));
}

void PrintDryRunState(double elapsed, const LowStateMsg& msg, const LoopStats& stats) {
  const auto q = ReadQ(msg);
  const auto dq = ReadDq(msg);
  const auto quat = ReadImuQuatWxyz(msg);
  const auto rpy = QuatWxyzToRpy(quat);
  const auto gyro = ReadGyro(msg);
  std::cout << "[LEGBOT-RT][DRY-RUN] t=" << std::fixed << std::setprecision(2) << elapsed
            << " tick=" << msg.tick()
            << " lowstate_count=" << stats.lowstate_count
            << " lost=" << LostFlagsText(msg)
            << " tilt=" << std::setprecision(5) << TiltFromLowState(msg) << "\n"
            << "  q=" << ArrayText(q, 4) << "\n"
            << "  dq=" << ArrayText(dq, 4) << "\n"
            << "  imu_quat_wxyz=" << ArrayText(quat, 5) << "\n"
            << "  imu_rpy=" << ArrayText(rpy, 5) << " gyro=" << ArrayText(gyro, 5) << "\n";
}

void RunDryRun(const Args& args, const std::shared_ptr<LowStateSubscriber>& sub, LoopStats* stats,
               const Clock::time_point& stats_t0) {
  WaitForAnyLowState(sub, args);
  std::cout << "[LEGBOT-RT][DRY-RUN] LowState received. LowCmd publisher was not created; rt/lowcmd will not be published.\n";
  const auto start = Clock::now();
  auto next = start;
  auto next_print = start;
  Clock::time_point last_sample_t = start;
  uint32_t last_tick = std::numeric_limits<uint32_t>::max();
  const double period_s = 1.0 / std::max(args.cmd_hz, 1.0);

  while (g_running) {
    const auto now = Clock::now();
    const double elapsed = SecondsSince(start, now);
    if (elapsed >= args.duration_s) break;
    if (sub->isTimeout()) throw std::runtime_error("rt_lowstate_timeout during dry-run");

    const auto msg = SnapshotLowState(sub);
    RecordLowStateSample(msg, stats, &last_sample_t, &last_tick);
    ++stats->loop_count;

    if (now >= next_print || stats->loop_count == 1) {
      PrintDryRunState(SecondsSince(stats_t0, now), msg, *stats);
      next_print = now + std::chrono::seconds(1);
    }

    next += std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(period_s));
    std::this_thread::sleep_until(next);
  }
  const auto final_msg = SnapshotLowState(sub);
  PrintDryRunState(SecondsSince(stats_t0, Clock::now()), final_msg, *stats);
  std::cout << "[LEGBOT-RT][DRY-RUN] completed; no LowCmd publisher created and no LowCmd published.\n";
}


void PrintFeedbackBootstrapState(double elapsed, const LowStateMsg& msg, const LoopStats& stats) {
  const auto q = ReadQ(msg);
  const auto dq = ReadDq(msg);
  const auto quat = ReadImuQuatWxyz(msg);
  const auto rpy = QuatWxyzToRpy(quat);
  const auto gyro = ReadGyro(msg);
  std::cout << "[LEGBOT-RT][FEEDBACK-BOOTSTRAP] t=" << std::fixed << std::setprecision(2) << elapsed
            << " tick=" << msg.tick()
            << " lowstate_count=" << stats.lowstate_count
            << " publish_count=" << stats.publish_count
            << " lost=" << LostFlagsText(msg)
            << " tilt=" << std::setprecision(5) << TiltFromLowState(msg) << "\n"
            << "  q=" << ArrayText(q, 4) << "\n"
            << "  dq=" << ArrayText(dq, 4) << "\n"
            << "  imu_quat_wxyz=" << ArrayText(quat, 5) << "\n"
            << "  imu_rpy=" << ArrayText(rpy, 5) << " gyro=" << ArrayText(gyro, 5) << "\n";
}

void RunFeedbackBootstrap(const Args& args, const std::shared_ptr<LowStateSubscriber>& sub,
                          LowCmdPublisher& pub, LoopStats* stats, const Clock::time_point& stats_t0) {
  WaitForAnyLowState(sub, args);
  std::cout << "[LEGBOT-RT][FEEDBACK-BOOTSTRAP] LowState received. Publishing mode=1 with kp=0, kd=0, tau=0 only; "
            << "no stand ramp, no IK, no trot, no lost gate during bootstrap.\n";

  const auto start = Clock::now();
  auto next = start;
  auto next_print = start;
  Clock::time_point last_sample_t = start;
  uint32_t last_tick = std::numeric_limits<uint32_t>::max();
  const double period_s = 1.0 / std::max(args.cmd_hz, 1.0);

  LowStateMsg final_msg = SnapshotLowState(sub);
  bool final_has_lost = true;
  bool printed_valid_transition = false;

  while (g_running) {
    const auto now = Clock::now();
    const double elapsed = SecondsSince(start, now);
    if (elapsed >= args.duration_s) break;
    if (sub->isTimeout()) throw std::runtime_error("rt_lowstate_timeout during feedback-bootstrap");

    const auto loop_t0 = Clock::now();
    const auto msg = SnapshotLowState(sub);
    final_msg = msg;
    final_has_lost = HasLostMotor(msg, nullptr);
    RecordLowStateSample(msg, stats, &last_sample_t, &last_tick);

    const auto q_hold = ReadQ(msg);
    if (!PublishLowCmd(pub, 1, q_hold, 0.0, 0.0, stats, stats_t0)) {
      throw std::runtime_error("publish_failed during feedback-bootstrap");
    }

    ++stats->loop_count;
    const double loop_ms = std::chrono::duration<double, std::milli>(Clock::now() - loop_t0).count();
    stats->loop_ms_sum += loop_ms;
    stats->loop_ms_max = std::max(stats->loop_ms_max, loop_ms);

    if (now >= next_print || stats->loop_count == 1 || (!final_has_lost && !printed_valid_transition)) {
      PrintFeedbackBootstrapState(SecondsSince(stats_t0, now), msg, *stats);
      if (!final_has_lost) printed_valid_transition = true;
      next_print = now + std::chrono::seconds(1);
    }

    next += std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(period_s));
    std::this_thread::sleep_until(next);
  }

  if (!g_running) return;
  final_msg = SnapshotLowState(sub);
  final_has_lost = HasLostMotor(final_msg, nullptr);
  PrintFeedbackBootstrapState(SecondsSince(stats_t0, Clock::now()), final_msg, *stats);

  if (final_has_lost) {
    throw std::runtime_error("feedback_bootstrap_failed:lost=" + LostFlagsText(final_msg));
  }
  std::cout << "[LEGBOT-RT][FEEDBACK-BOOTSTRAP] feedback_bootstrap_ok: lost=none.\n";
}

template <typename TargetFn>
void RunTimedPhase(const char* phase_name, double seconds, double hz, const Args& args,
                   const std::shared_ptr<LowStateSubscriber>& sub, LowCmdPublisher& pub, LoopStats* stats,
                   const Clock::time_point& stats_t0, TargetFn target_fn) {
  if (seconds <= 0.0) return;
  const double period_s = 1.0 / hz;
  const auto start = Clock::now();
  auto next = start;
  while (g_running) {
    const auto now = Clock::now();
    const double elapsed = SecondsSince(start, now);
    if (elapsed >= seconds) break;
    if (sub->isTimeout()) throw std::runtime_error(std::string("rt_lowstate_timeout during ") + phase_name);
    const auto loop_t0 = Clock::now();
    const auto msg = SnapshotLowState(sub);
    const auto q_target = target_fn(elapsed, msg);
    CheckSafety(msg, q_target, args, stats);
    if (!PublishLowCmd(pub, 1, q_target, args.kp, args.kd, stats, stats_t0)) {
      throw std::runtime_error(std::string("publish_failed during ") + phase_name);
    }
    ++stats->loop_count;
    const double loop_ms = std::chrono::duration<double, std::milli>(Clock::now() - loop_t0).count();
    stats->loop_ms_sum += loop_ms;
    stats->loop_ms_max = std::max(stats->loop_ms_max, loop_ms);
    next += std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(period_s));
    std::this_thread::sleep_until(next);
  }
}


void RunReturnDownOnExit(double seconds, const Args& args,
                         const std::shared_ptr<LowStateSubscriber>& sub, LowCmdPublisher& pub,
                         LoopStats* stats, const Clock::time_point& stats_t0) {
  if (seconds <= 0.0) return;

  const auto start_msg = SnapshotLowState(sub);
  const auto q_start = ReadQ(start_msg);
  const double hz = std::max(args.cmd_hz, 1.0);
  const double period_s = 1.0 / hz;
  const auto start = Clock::now();
  auto next = start;

  std::cout << "[LEGBOT-RT] normal exit: ramp current q -> down for "
            << seconds << "s, then disable\n";

  while (true) {
    const auto now = Clock::now();
    const double elapsed = SecondsSince(start, now);
    if (elapsed >= seconds) break;
    if (sub->isTimeout()) throw std::runtime_error("rt_lowstate_timeout during return_down");

    const auto loop_t0 = Clock::now();
    const auto msg = SnapshotLowState(sub);
    const double alpha = SmoothStep(elapsed / std::max(seconds, 1.0e-9));
    std::array<double, kNumJoints> q{};
    for (size_t i = 0; i < kNumJoints; ++i) {
      q[i] = (1.0 - alpha) * q_start[i] + alpha * kDownQ[i];
    }

    const auto q_now = ReadQ(msg);
    stats->max_qerr = std::max(stats->max_qerr, MaxQError(q_now, q));
    stats->max_tilt = std::max(stats->max_tilt, TiltFromLowState(msg));

    if (!PublishLowCmd(pub, 1, q, args.kp, args.kd, stats, stats_t0)) {
      throw std::runtime_error("publish_failed during return_down");
    }

    ++stats->loop_count;
    const double loop_ms = std::chrono::duration<double, std::milli>(Clock::now() - loop_t0).count();
    stats->loop_ms_sum += loop_ms;
    stats->loop_ms_max = std::max(stats->loop_ms_max, loop_ms);

    next += std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(period_s));
    std::this_thread::sleep_until(next);
  }

  // Hold the final down pose briefly for one control tick before disabling.
  PublishLowCmd(pub, 1, kDownQ, args.kp, args.kd, stats, stats_t0);
}

void PrintSummary(const std::string& exit_reason, double elapsed_s, const LoopStats& stats) {
  const auto& times = stats.publish_times_s;
  double mean = std::numeric_limits<double>::quiet_NaN();
  double min_v = std::numeric_limits<double>::quiet_NaN();
  double max_v = std::numeric_limits<double>::quiet_NaN();
  if (times.size() >= 2) {
    double sum = 0.0;
    min_v = std::numeric_limits<double>::infinity();
    max_v = 0.0;
    for (size_t i = 1; i < times.size(); ++i) {
      const double dt = times[i] - times[i - 1];
      sum += dt;
      min_v = std::min(min_v, dt);
      max_v = std::max(max_v, dt);
    }
    mean = sum / static_cast<double>(times.size() - 1);
  }
  const double loop_ms_mean =
      stats.loop_count > 0 ? stats.loop_ms_sum / static_cast<double>(stats.loop_count)
                           : std::numeric_limits<double>::quiet_NaN();

  std::cout << std::fixed << std::setprecision(6)
            << "exit_reason=" << exit_reason << "\n"
            << "elapsed_s=" << elapsed_s << "\n"
            << "loop_count=" << stats.loop_count << "\n"
            << "publish_count=" << stats.publish_count << "\n"
            << "lowstate_count=" << stats.lowstate_count << "\n"
            << "lowstate_interval_mean_s="
            << (stats.lowstate_count >= 2 ? stats.lowstate_interval_sum_s / static_cast<double>(stats.lowstate_count - 1)
                                          : std::numeric_limits<double>::quiet_NaN())
            << "\n"
            << "lowstate_interval_min_s="
            << (stats.lowstate_count >= 2 ? stats.lowstate_interval_min_s : std::numeric_limits<double>::quiet_NaN())
            << "\n"
            << "lowstate_interval_max_s="
            << (stats.lowstate_count >= 2 ? stats.lowstate_interval_max_s : std::numeric_limits<double>::quiet_NaN())
            << "\n"
            << "publish_interval_mean_s=" << mean << "\n"
            << "publish_interval_min_s=" << min_v << "\n"
            << "publish_interval_max_s=" << max_v << "\n"
            << "loop_ms_mean=" << loop_ms_mean << "\n"
            << "loop_ms_max=" << stats.loop_ms_max << "\n"
            << "max_qerr=" << stats.max_qerr << "\n"
            << "max_tilt=" << stats.max_tilt << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, OnSignal);
  std::signal(SIGTERM, OnSignal);

  Args args;
  std::unique_ptr<LowCmdPublisher> lowcmd_pub;
  LoopStats stats;
  const auto stats_t0 = Clock::now();
  std::string exit_reason = "not_started";
  double elapsed_s = 0.0;

  try {
    args = ParseArgs(argc, argv);
    if (args.help) {
      PrintHelp(argv[0]);
      return 0;
    }
    ValidateArgs(args);

    if (args.dry_run) {
      std::cout << "[LEGBOT-RT] DDS topics: subscribe " << kLowStateTopic
                << "; DRY-RUN: no LowCmd publisher, no " << kLowCmdTopic << " publish\n";
    } else if (args.feedback_bootstrap) {
      std::cout << "[LEGBOT-RT] DDS topics: subscribe " << kLowStateTopic
                << ", publish " << kLowCmdTopic
                << "; FEEDBACK-BOOTSTRAP: zero kp/kd/tau only\n";
    } else {
      std::cout << "[LEGBOT-RT] DDS topics: subscribe " << kLowStateTopic << ", publish " << kLowCmdTopic << "\n";
    }
    std::cout << "[LEGBOT-RT] mode=" << args.mode << " cmd_hz=" << args.cmd_hz
              << " gait_hz=" << args.gait_frequency_hz << " duty=" << args.gait_duty
              << " step_length=" << args.step_length << " tau=0"
              << (args.return_down_on_exit ? " return_down_on_exit=1" : "")
              << (args.dry_run ? " DRY-RUN" : "")
              << (args.feedback_bootstrap ? " FEEDBACK-BOOTSTRAP" : "") << "\n";

    unitree::robot::ChannelFactory::Instance()->Init(0, args.network);
    auto lowstate_sub = std::make_shared<LowStateSubscriber>(kLowStateTopic);
    lowstate_sub->set_timeout_ms(static_cast<uint32_t>(std::ceil(args.lowstate_timeout_s * 1000.0)));

    if (args.dry_run) {
      RunDryRun(args, lowstate_sub, &stats, stats_t0);
      elapsed_s = SecondsSince(stats_t0, Clock::now());
      exit_reason = g_running ? "dry_run_completed" : "signal";
      PrintSummary(exit_reason, elapsed_s, stats);
      return 0;
    }

    lowcmd_pub = std::make_unique<LowCmdPublisher>(kLowCmdTopic);

    if (args.feedback_bootstrap) {
      RunFeedbackBootstrap(args, lowstate_sub, *lowcmd_pub, &stats, stats_t0);
      elapsed_s = SecondsSince(stats_t0, Clock::now());
      exit_reason = g_running ? "feedback_bootstrap_ok" : "signal";
      PublishDisableBurst(lowcmd_pub.get(), args, &stats, stats_t0);
      PrintSummary(exit_reason, elapsed_s, stats);
      return exit_reason == "feedback_bootstrap_ok" || exit_reason == "signal" ? 0 : 1;
    }

    WaitForValidLowState(lowstate_sub, args);
    auto first_msg = SnapshotLowState(lowstate_sub);
    const auto q_start = ReadQ(first_msg);
    std::array<double, kNumJoints> q_target_prev = q_start;

    const auto stand_feet = StandFeetBody();
    std::cout << "[LEGBOT-RT] LowState valid; ramp current q -> stand for "
              << args.startup_ramp_seconds << "s, prehold " << args.prehold_seconds << "s\n";

    RunTimedPhase("startup_ramp", args.startup_ramp_seconds, args.cmd_hz, args, lowstate_sub, *lowcmd_pub,
                  &stats, stats_t0, [&](double elapsed, const LowStateMsg&) {
                    const double alpha = SmoothStep(elapsed / std::max(args.startup_ramp_seconds, 1.0e-9));
                    std::array<double, kNumJoints> q{};
                    for (size_t i = 0; i < kNumJoints; ++i) {
                      q[i] = (1.0 - alpha) * q_start[i] + alpha * kStandQ[i];
                    }
                    q_target_prev = q;
                    return q;
                  });

    q_target_prev = kStandQ;
    RunTimedPhase("prehold", args.prehold_seconds, args.cmd_hz, args, lowstate_sub, *lowcmd_pub, &stats,
                  stats_t0, [&](double, const LowStateMsg&) { return kStandQ; });

    auto prev_contact_mask = std::array<int, kNumLegs>{1, 1, 1, 1};
    const auto control_start = Clock::now();
    auto next = control_start;
    const double period_s = 1.0 / args.cmd_hz;
    while (g_running) {
      const auto now = Clock::now();
      const double gait_time_s = SecondsSince(control_start, now);
      elapsed_s = SecondsSince(stats_t0, now);
      if (gait_time_s >= args.duration_s) {
        exit_reason = "completed_duration";
        break;
      }
      if (lowstate_sub->isTimeout()) throw std::runtime_error("rt_lowstate_timeout");

      const auto loop_t0 = Clock::now();
      const auto msg = SnapshotLowState(lowstate_sub);
      const auto q_now = ReadQ(msg);
      std::array<int, kNumLegs> contact_mask{};
      auto q_raw = ComputeRawTarget(args, gait_time_s, stand_feet, q_target_prev, &contact_mask);
      auto q_target = RateLimitTargets(q_raw, q_target_prev, q_now, contact_mask, prev_contact_mask, args);
      ApplySoftLimits(&q_target);
      CheckSafety(msg, q_target, args, &stats);

      if (!PublishLowCmd(*lowcmd_pub, 1, q_target, args.kp, args.kd, &stats, stats_t0)) {
        throw std::runtime_error("publish_failed");
      }

      q_target_prev = q_target;
      prev_contact_mask = contact_mask;
      ++stats.loop_count;
      const double loop_ms = std::chrono::duration<double, std::milli>(Clock::now() - loop_t0).count();
      stats.loop_ms_sum += loop_ms;
      stats.loop_ms_max = std::max(stats.loop_ms_max, loop_ms);

      static uint64_t print_div = 0;
      const uint64_t print_every = static_cast<uint64_t>(std::max(1.0, args.cmd_hz));
      if ((++print_div % print_every) == 0) {
        std::cout << "[LEGBOT-RT] t=" << std::fixed << std::setprecision(2) << gait_time_s
                  << " mask=" << MaskText(contact_mask)
                  << " max_qerr=" << std::setprecision(4) << stats.max_qerr
                  << " max_tilt=" << stats.max_tilt << "\n";
      }

      next += std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(period_s));
      std::this_thread::sleep_until(next);
    }

    if (!g_running && exit_reason != "completed_duration") exit_reason = "signal";
    if (exit_reason == "completed_duration" && args.return_down_on_exit) {
      RunReturnDownOnExit(args.return_ramp_seconds, args, lowstate_sub, *lowcmd_pub, &stats, stats_t0);
      PublishDisableFor(lowcmd_pub.get(), args, &stats, stats_t0, args.exit_disable_seconds);
    } else if (args.disable_on_exit || exit_reason != "completed_duration") {
      PublishDisableFor(lowcmd_pub.get(), args, &stats, stats_t0, args.exit_disable_seconds);
    }
    PrintSummary(exit_reason, SecondsSince(stats_t0, Clock::now()), stats);
    return exit_reason == "completed_duration" || exit_reason == "signal" ? 0 : 1;
  } catch (const std::exception& e) {
    exit_reason = std::string("exception:") + e.what();
    elapsed_s = SecondsSince(stats_t0, Clock::now());
    try {
      PublishDisableBurst(lowcmd_pub.get(), args, &stats, stats_t0);
    } catch (...) {
    }
    PrintSummary(exit_reason, elapsed_s, stats);
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}

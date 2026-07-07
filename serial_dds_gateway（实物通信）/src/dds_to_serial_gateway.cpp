// Fatu Phase-1 DDS 串口网关：订阅 rt/lowcmd，发布 rt/lowstate；
// 经灵足 USB-CAN 串口驱动 12 路电机，可选独立 IMU 串口。

#include "imu_framer.hpp"
#include "imu_gyro_filter.hpp"
#include "joint_motor_bias.hpp"
#include "lingzu_serial.hpp"
#include "lingzu_motor_protocol.hpp"
#include "motor_map.hpp"
#include "protocol_codec.hpp"
#include "serial_framer.hpp"

#include <atomic>
#include <algorithm>
#include <array>
#include <chrono>
#include <csignal>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "unitree/robot/channel/channel_factory.hpp"
#include "unitree/dds_wrapper/robots/go2/go2_pub.h"
#include "unitree/dds_wrapper/robots/go2/go2_sub.h"
#include <unitree/idl/go2/LowCmd_.hpp>

using namespace serial_dds_gateway;

namespace {

// SIGINT/SIGTERM 时置 false，主循环与 RX 线程退出
std::atomic<bool> g_running{true};

constexpr const char* kDefaultJointBiasFile = "config/joint_prone_bias.fatu.txt";

// 单路电机 type2 反馈（电机编码器空间）
struct MotorFeedbackCache {
  float q{0.0f};
  float dq{0.0f};
  float tau_est{0.0f};
  uint8_t temperature{0};
  bool seen{false};  // 是否至少收到过一帧 type2
};

// IMU 解析结果（四元数 w,x,y,z + 滤波后陀螺仪）
struct ImuCache {
  std::array<float, 4> quaternion{1.0F, 0.0F, 0.0F, 0.0F};
  std::array<float, 3> gyroscope{0.0F, 0.0F, 0.0F};
  bool seen{false};
};

// 运行统计，[STAT] 每秒打印；多线程通过 atomic 更新
struct GatewayStats {
  std::atomic<uint64_t> rx_frames{0};
  std::atomic<uint64_t> rx_a_frames{0};
  std::atomic<uint64_t> rx_b_frames{0};
  std::atomic<uint64_t> type2_frames{0};
  std::atomic<uint64_t> type2_a_frames{0};
  std::atomic<uint64_t> type2_b_frames{0};
  std::atomic<uint64_t> decode_errors{0};
  std::atomic<uint64_t> tx_type1_frames{0};
  std::atomic<uint64_t> tx_enable_frames{0};
  std::atomic<uint64_t> tx_disable_frames{0};
  std::atomic<uint64_t> tx_a_frames{0};
  std::atomic<uint64_t> tx_b_frames{0};
  std::atomic<uint64_t> imu_frames{0};
  std::atomic<uint64_t> imu_errors{0};
  std::atomic<uint64_t> tx_write_errors{0};  // 串口 write 失败（如 USB 掉线 EIO）
};

// 析构时 join 后台线程
struct RunningThreadGuard {
  std::thread& thread;
  explicit RunningThreadGuard(std::thread& t) : thread(t) {}
  ~RunningThreadGuard() {
    g_running = false;
    if (thread.joinable()) thread.join();
  }
};

void OnSignal(int) { g_running = false; }

uint8_t JointCanId(std::string_view joint) {
  const auto it = kJointToCanId.find(joint);
  if (it == kJointToCanId.end()) throw std::runtime_error("joint not in map");
  return it->second;
}

// 命令行配置
struct Args {
  std::string serial_port;       // 单串口（legacy）
  std::string serial_port_a;     // 双串口 A：FR+RR
  std::string serial_port_b;     // 双串口 B：FL+RL
  int baudrate = 2000000;       // 2M 是默认值，也可以使用 921600 或 115200
  std::string imu_port;
  int imu_baudrate = 921600;   // 921600 是默认值，也可以使用 2000000 或 115200
  std::string network = "lo";    // DDS 网卡，实机与 fatu_ctrl 一致用 lo
  uint8_t channel = 0x00;        // 灵足帧 channel 字节
  uint16_t master_id = 0x00FD;
  double tick_hz = 500.0;        // 主循环：发 type1 + 发布 lowstate 的频率
  bool enable_on_start = false;  // 启动即 type3 使能（实机一般由 FSM 控制）
  bool disable_on_exit = false;  // Ctrl+C 时 type4 失能 12 路
  bool wait_lowcmd = false;      // 等 lowcmd 再发 lowstate（实机易与 fatu_ctrl 死锁）
  bool imu_gyro_calib = true;   // 是否启用 IMU 陀螺仪静止偏置标定
  double imu_gyro_calib_seconds = 2.0;  // 静止偏置标定时间
  double imu_gyro_deadzone = 0.1;  // 静止偏置标定死区 (rad/s)
  bool joint_bias_calib = false;  // 上电在线标定趴姿 joint bias
  double joint_bias_calib_seconds = 2.0;  // 在线标定时间
  double joint_bias_calib_timeout_seconds = 60.0;  // 在线标定超时时间
  std::string joint_bias_reference;
  std::string joint_bias_reference_file;  // 在线标定参考文件
  std::string joint_bias_load_file = kDefaultJointBiasFile;  // 加载已保存 bias 文件
};

// 解析命令行参数，填充 Args；末尾校验串口模式互斥
Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    // 读取当前选项的下一个参数；缺失则抛异常
    auto need = [&](const char* name) {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return std::string(argv[++i]);
    };

    // --- 串口与总线 ---
    if (k == "--serial-port") a.serial_port = need("--serial-port");
    else if (k == "--serial-port-a") a.serial_port_a = need("--serial-port-a");
    else if (k == "--serial-port-b") a.serial_port_b = need("--serial-port-b");
    else if (k == "--baudrate") a.baudrate = std::stoi(need("--baudrate"));
    else if (k == "--imu-port") a.imu_port = need("--imu-port");
    else if (k == "--imu-baudrate") a.imu_baudrate = std::stoi(need("--imu-baudrate"));

    // --- DDS 与灵足帧参数 ---
    else if (k == "--network") a.network = need("--network");
    else if (k == "--channel") a.channel = static_cast<uint8_t>(std::stoi(need("--channel"), nullptr, 0));
    else if (k == "--master-id") a.master_id = static_cast<uint16_t>(std::stoi(need("--master-id"), nullptr, 0));
    else if (k == "--tick-hz") a.tick_hz = std::stod(need("--tick-hz"));

    // --- 启停行为（布尔开关，无额外参数）---
    else if (k == "--send-enable-on-start") a.enable_on_start = true;
    else if (k == "--send-disable-on-exit") a.disable_on_exit = true;
    else if (k == "--wait-lowcmd") a.wait_lowcmd = true;

    // --- IMU 陀螺滤波 ---
    else if (k == "--no-imu-gyro-calib") a.imu_gyro_calib = false;
    else if (k == "--imu-gyro-calib-seconds") a.imu_gyro_calib_seconds = std::stod(need("--imu-gyro-calib-seconds"));
    else if (k == "--imu-gyro-deadzone") a.imu_gyro_deadzone = std::stod(need("--imu-gyro-deadzone"));

    // --- 关节 bias：在线标定或加载文件 ---
    else if (k == "--no-joint-bias-calib") a.joint_bias_calib = false;
    else if (k == "--joint-bias-calib") a.joint_bias_calib = true;
    else if (k == "--joint-bias-calib-seconds")
      a.joint_bias_calib_seconds = std::stod(need("--joint-bias-calib-seconds"));
    else if (k == "--joint-bias-calib-timeout")
      a.joint_bias_calib_timeout_seconds = std::stod(need("--joint-bias-calib-timeout"));
    else if (k == "--joint-bias-reference") a.joint_bias_reference = need("--joint-bias-reference");
    else if (k == "--joint-bias-reference-file")
      a.joint_bias_reference_file = need("--joint-bias-reference-file");
    else if (k == "--joint-bias-load-file") a.joint_bias_load_file = need("--joint-bias-load-file");
    else if (k == "--no-joint-bias-file") a.joint_bias_load_file.clear();
  }

  // 串口模式三选一：单口 legacy，或双口 a+b，不可混用
  const bool has_legacy = !a.serial_port.empty();
  const bool has_a = !a.serial_port_a.empty();
  const bool has_b = !a.serial_port_b.empty();
  if (has_legacy && (has_a || has_b)) {
    throw std::runtime_error("use either --serial-port or both --serial-port-a/--serial-port-b, not both");
  }
  if (has_a != has_b) {
    throw std::runtime_error("use both --serial-port-a and --serial-port-b for dual motor serial mode");
  }
  if (!has_legacy && !has_a) {
    throw std::runtime_error(
        "use --serial-port /dev/myttyCAN0 or --serial-port-a /dev/myttyCAN0 --serial-port-b /dev/myttyCAN1");
  }
  return a;
}

// type3 使能 / type4 失能
SerialFrame BuildType34(uint8_t mode, uint8_t motor_id, uint8_t master_id, uint8_t channel) {
  return BuildMotorModeFrame(channel, master_id, motor_id, mode);
}

void AddRxFrames(GatewayStats& stats, MotorSerialBus bus, size_t count) {
  stats.rx_frames.fetch_add(count);
  if (bus == MotorSerialBus::A) {
    stats.rx_a_frames.fetch_add(count);
  } else {
    stats.rx_b_frames.fetch_add(count);
  }
}

void AddType2Frame(GatewayStats& stats, MotorSerialBus bus) {
  ++stats.type2_frames;
  if (bus == MotorSerialBus::A) {
    ++stats.type2_a_frames;
  } else {
    ++stats.type2_b_frames;
  }
}

void AddTxFrame(GatewayStats& stats, MotorSerialBus bus) {
  if (bus == MotorSerialBus::A) {
    ++stats.tx_a_frames;
  } else {
    ++stats.tx_b_frames;
  }
}

// 电机串口接收线程：与 tick_hz 无关，约每 1ms 轮询；解码 type2 写入 motor_cache
void RxLoop(SerialFramer& framer, const RangeSpec& ranges, const std::unordered_map<uint8_t, size_t>& canid_to_index,
            std::array<MotorFeedbackCache, 12>& motor_cache, std::mutex& cache_mutex, GatewayStats& stats,
            MotorSerialBus bus) {
  while (g_running) {
    //读串口并拆帧，返回帧列表
    const auto rx_frames = framer.ReadAvailableFrames();
    AddRxFrames(stats, bus, rx_frames.size());
    //遍历帧列表，处理每帧
    for (const auto& f : rx_frames) {
      //帧数据长度小于8字节，跳过
      if (f.data.size() < 8) {
        ++stats.decode_errors;
        continue;
      }
      //解码帧数据，返回Type2Feedback结构体
      Type2Feedback fb; //解码得到 Type2Feedback：motor_id、q、dq、tau、temp_c 等基本信息
      try {
        fb = DecodeType2SerialFrame(f, ranges);
      } catch (const std::exception&) {
        ++stats.decode_errors;
        continue;
      }
      //根据 motor_id 查找 kJointOrder 下标
      auto map_it = canid_to_index.find(fb.motor_id);
      //如果找不到，跳过
      if (map_it == canid_to_index.end()) {
        ++stats.decode_errors;
        continue;
      }
      {
        std::lock_guard<std::mutex> lock(cache_mutex);
        // 加锁保护 motor_cache 访问
        auto& cached = motor_cache[map_it->second];  // 下标对应 kJointOrder：FR,FL,RR,RL
        // 更新 motor_cache 中对应关节的 q、dq、tau、temp_c 信息
        cached.q = static_cast<float>(fb.q);
        cached.dq = static_cast<float>(fb.dq);
        cached.tau_est = static_cast<float>(fb.tau);
        cached.temperature = static_cast<uint8_t>(std::max(0.0, std::min(255.0, fb.temp_c)));
        cached.seen = true;
      }
      // 更新统计信息：帧计数 + 总线分类
      AddType2Frame(stats, bus);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

// IMU 串口接收线程：欧拉角→四元数，陀螺仪经静止偏置标定与死区
void ImuRxLoop(ImuFramer& framer, ImuGyroFilter& gyro_filter, ImuCache& imu_cache, std::mutex& imu_mutex,
               GatewayStats& stats) {
  while (g_running) {
    try {
      const auto samples = framer.ReadAvailableSamples();
      for (const auto& sample : samples) {
        const auto q = EulerYprToQuaternion(sample.yaw, sample.pitch, sample.roll);
        const auto gyro = gyro_filter.Apply(static_cast<float>(sample.gx), static_cast<float>(sample.gy),
                                            static_cast<float>(sample.gz));
        {
          std::lock_guard<std::mutex> lock(imu_mutex);
          imu_cache.quaternion = {static_cast<float>(q.w), static_cast<float>(q.x), static_cast<float>(q.y),
                                  static_cast<float>(q.z)};
          imu_cache.gyroscope = gyro;
          imu_cache.seen = true;
        }
        ++stats.imu_frames;
      }
    } catch (const std::exception&) {
      ++stats.imu_errors;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

}  // namespace

void PrintPhase1Banner(const Args& args, bool dual_motor_serial) {
  std::cout << "\n========== Fatu Phase-1: DDS Serial Gateway ==========\n";
  std::cout << "[PHASE1] Orange Pi onboard control, DDS network=" << args.network << "\n";
  if (dual_motor_serial) {
    std::cout << "[PHASE1] motor serial A (FR/RR): " << args.serial_port_a << " @" << args.baudrate << "\n";
    std::cout << "[PHASE1] motor serial B (FL/RL): " << args.serial_port_b << " @" << args.baudrate << "\n";
  } else {
    std::cout << "[PHASE1] motor serial: " << args.serial_port << " @" << args.baudrate << "\n";
  }
  if (!args.imu_port.empty()) {
    std::cout << "[PHASE1] IMU serial: " << args.imu_port << " @" << args.imu_baudrate << "\n";
    if (args.imu_gyro_calib) {
      std::cout << "[PHASE1] IMU scheme A: gateway gyro bias " << args.imu_gyro_calib_seconds
                << "s (keep still), deadzone=" << args.imu_gyro_deadzone << " rad/s\n";
    } else {
      std::cout << "[PHASE1] IMU gyro calibration disabled\n";
    }
  } else {
    std::cout << "[PHASE1] IMU serial: disabled (identity quaternion published)\n";
  }
  std::cout << "[PHASE1] DDS topics: subscribe rt/lowcmd, publish rt/lowstate\n";
  std::cout << "[PHASE1] tick_hz=" << args.tick_hz << " channel=0x" << std::hex << static_cast<int>(args.channel)
            << std::dec << " master_id=0x" << std::hex << args.master_id << std::dec << "\n";
  if (args.enable_on_start) {
    std::cout << "[PHASE1] send-enable-on-start: ON (FSM mode not required)\n";
  } else {
    std::cout << "[PHASE1] motor enable: from fatu_ctrl when motor_cmd.mode != 0\n";
  }
  if (args.disable_on_exit) {
    std::cout << "[PHASE1] send-disable-on-exit: ON (Ctrl+C sends type4 to all motors)\n";
  }
  std::cout << "[PHASE1] joint map: q/dq calf 2:1 gear; kp/kd/tau 1:1; sign (FL/RL thigh+calf=-1)\n";
  if (args.joint_bias_calib) {
    std::cout << "[PHASE1] joint bias: online prone calib " << args.joint_bias_calib_seconds
              << "s (use --no-joint-bias-calib + bias file for production)\n";
  } else if (!args.joint_bias_load_file.empty()) {
    std::cout << "[PHASE1] joint bias: load " << args.joint_bias_load_file
              << " (no online calib at power-on)\n";
  } else {
    std::cout << "[PHASE1] joint bias: builtin kFatuProneBiasFromLog (no online calib)\n";
  }
  std::cout << "[PHASE1] next: start terminal 2 -> fatu_ctrl --network lo\n";
  std::cout << "====================================================\n\n";
}

int main(int argc, char** argv) {
  std::signal(SIGINT, OnSignal);
  std::signal(SIGTERM, OnSignal);
  try {
    const auto args = ParseArgs(argc, argv);
    RangeSpec ranges;  // type1/type2 量化范围
    const bool dual_motor_serial = !args.serial_port_a.empty();
    std::unique_ptr<SerialFramer> framer_a;
    std::unique_ptr<SerialFramer> framer_b;
    if (dual_motor_serial) {
      framer_a = std::make_unique<SerialFramer>(args.serial_port_a, args.baudrate);
      framer_b = std::make_unique<SerialFramer>(args.serial_port_b, args.baudrate);
      std::cout << "[INFO] motor serial A(FR/RR)=" << args.serial_port_a << " B(FL/RL)=" << args.serial_port_b
                << " @" << args.baudrate << "\n";
    } else {
      framer_a = std::make_unique<SerialFramer>(args.serial_port, args.baudrate);
      std::cout << "[INFO] motor serial single=" << args.serial_port << " @" << args.baudrate << "\n";
    }
    std::unique_ptr<ImuFramer> imu_framer;
    ImuGyroFilter imu_gyro_filter;
    if (!args.imu_port.empty()) {
      imu_framer = std::make_unique<ImuFramer>(args.imu_port, args.imu_baudrate);
      imu_gyro_filter.Configure(args.imu_gyro_calib, args.imu_gyro_calib_seconds, args.imu_gyro_deadzone);
      std::cout << "[INFO] IMU serial enabled on " << args.imu_port << " @" << args.imu_baudrate << "\n";
    }

    PrintPhase1Banner(args, dual_motor_serial);

    unitree::robot::ChannelFactory::Instance()->Init(0, args.network);
    std::cout << "[PHASE1] DDS ChannelFactory initialized on \"" << args.network << "\"\n";

    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>("rt/lowcmd");
    auto lowstate_pub = std::make_unique<unitree::robot::go2::publisher::LowState>("rt/lowstate");
    lowstate_pub->msg_.head() = {0xFE, 0xEF};
    lowstate_pub->msg_.level_flag() = 0xFF;

    if (args.wait_lowcmd) {
      std::cout << "[PHASE1] waiting for rt/lowcmd publisher...\n";
      lowcmd_sub->wait_for_connection();
      std::cout << "[PHASE1] rt/lowcmd publisher connected\n";
    } else {
      std::cout << "[PHASE1] publishing rt/lowstate now (not waiting for rt/lowcmd)\n";
      std::cout << "[PHASE1] start fatu_ctrl in terminal 2 if not running yet\n";
    }

    GatewayStats stats;
    auto motor_bus = [&](uint8_t can_id) {
      return dual_motor_serial ? MotorSerialBusForCanId(can_id) : MotorSerialBus::A;
    };
    auto motor_framer = [&](uint8_t can_id) -> SerialFramer& {
      if (!dual_motor_serial) {
        return *framer_a;
      }
      return (MotorSerialBusForCanId(can_id) == MotorSerialBus::A) ? *framer_a : *framer_b;
    };
    // 写失败仅计数；EIO 时 SerialFramer 会自动重开串口
    auto write_motor_frame = [&](uint8_t can_id, const SerialFrame& frame) {
      const auto bus = motor_bus(can_id);
      if (!motor_framer(can_id).WriteFrame(frame)) {
        ++stats.tx_write_errors;
        return;
      }
      AddTxFrame(stats, bus);
    };

    if (args.enable_on_start) {
      for (auto j : kJointOrder) {
        const auto can_id = JointCanId(j);
        write_motor_frame(can_id, BuildType34(kLingzuMotorEnableCode, can_id, static_cast<uint8_t>(args.master_id),
                                              args.channel));
        ++stats.tx_enable_frames;
      }
    }

    // 检测 motor_cmd.mode 边沿，触发 type3/4
    std::unordered_map<uint8_t, bool> mode_nonzero_prev;
    for (auto j : kJointOrder) mode_nonzero_prev[JointCanId(j)] = false;
    // CAN 电机 ID → motor_cache 下标 [0..11]
    std::unordered_map<uint8_t, size_t> canid_to_index;
    for (size_t i = 0; i < kJointOrder.size(); ++i) {
      canid_to_index[JointCanId(kJointOrder[i])] = i;
    }
    std::array<MotorFeedbackCache, 12> motor_cache{};
    std::mutex cache_mutex;
    ImuCache imu_cache{};
    std::mutex imu_mutex;
    uint32_t tick = 0;
    std::thread rx_thread_a(RxLoop, std::ref(*framer_a), std::cref(ranges), std::cref(canid_to_index),
                            std::ref(motor_cache), std::ref(cache_mutex), std::ref(stats), MotorSerialBus::A);
    RunningThreadGuard rx_guard_a{rx_thread_a};
    std::unique_ptr<std::thread> rx_thread_b;
    std::unique_ptr<RunningThreadGuard> rx_guard_b;
    if (dual_motor_serial) {
      rx_thread_b = std::make_unique<std::thread>(RxLoop, std::ref(*framer_b), std::cref(ranges),
                                                  std::cref(canid_to_index), std::ref(motor_cache),
                                                  std::ref(cache_mutex), std::ref(stats), MotorSerialBus::B);
      rx_guard_b = std::make_unique<RunningThreadGuard>(*rx_thread_b);
    }
    std::unique_ptr<std::thread> imu_thread;
    std::unique_ptr<RunningThreadGuard> imu_guard;
    if (imu_framer) {
      imu_thread = std::make_unique<std::thread>(ImuRxLoop, std::ref(*imu_framer), std::ref(imu_gyro_filter),
                                                 std::ref(imu_cache), std::ref(imu_mutex), std::ref(stats));
      imu_guard = std::make_unique<RunningThreadGuard>(*imu_thread);
    }

    std::array<float, 12> joint_model_reference_q = kDefaultProneModelQ;
    if (!args.joint_bias_reference.empty()) {
      joint_model_reference_q = ParseJointBiasReference(args.joint_bias_reference);
    } else if (!args.joint_bias_reference_file.empty()) {
      joint_model_reference_q = LoadJointBiasReferenceFile(args.joint_bias_reference_file);
    }
    JointMotorBiasMap joint_bias;
    const bool apply_joint_bias = args.joint_bias_calib || !args.joint_bias_load_file.empty();
    joint_bias.Configure(apply_joint_bias, args.joint_bias_calib, joint_model_reference_q);
    if (args.joint_bias_calib) {
      std::cout << "[PHASE1] joint bias calib: keep prone/still; need 12/12 motor feedback for "
                << args.joint_bias_calib_seconds << "s (timeout "
                << args.joint_bias_calib_timeout_seconds << "s)\n";
    } else if (!args.joint_bias_load_file.empty()) {
      const std::string resolved = ResolveGatewayConfigPath(args.joint_bias_load_file);
      joint_bias.LoadBias(LoadJointBiasFile(args.joint_bias_load_file), resolved.c_str());
    } else if (apply_joint_bias) {
      joint_bias.LoadBias(FatuProneBiasFromLog(), "builtin kFatuProneBiasFromLog");
    }

    constexpr int kMinJointBiasSamples = 20;
    const auto dt = std::chrono::duration<double>(1.0 / std::max(1.0, args.tick_hz));
    while (g_running) {
      const auto t0 = std::chrono::steady_clock::now();

      // --- 1) 读取 fatu_ctrl 发布的 lowcmd ---
      unitree_go::msg::dds_::LowCmd_ cmd_snapshot;
      {
        std::lock_guard<std::mutex> lock(lowcmd_sub->mutex_);
        cmd_snapshot = lowcmd_sub->msg_;
      }

      // --- 2) lowcmd → 灵足串口：mode 边沿发 type3/4，使能后每 tick 逐电机发 type1 ---
      for (size_t i = 0; i < kJointOrder.size(); ++i) {
        const uint8_t can_id = JointCanId(kJointOrder[i]);
        const auto& m = cmd_snapshot.motor_cmd()[i];
        const bool mode_nonzero = (m.mode() != 0);
        if (mode_nonzero && !mode_nonzero_prev[can_id]) {
          write_motor_frame(can_id, BuildType34(kLingzuMotorEnableCode, can_id, static_cast<uint8_t>(args.master_id),
                                                args.channel));
          ++stats.tx_enable_frames;
        } else if (!mode_nonzero && mode_nonzero_prev[can_id]) {
          write_motor_frame(can_id, BuildType34(kLingzuMotorDisableCode, can_id, static_cast<uint8_t>(args.master_id),
                                                args.channel));
          ++stats.tx_disable_frames;
        }
        mode_nonzero_prev[can_id] = mode_nonzero;

        if (!mode_nonzero) {
          continue;
        }
        Type1Command t1{
            .motor_id = can_id,
            .q = joint_bias.ModelToMotor(i, m.q()),
            .dq = joint_bias.ModelToMotorDq(i, m.dq()),
            .kp = joint_bias.ModelToMotorKp(i, m.kp()),
            .kd = joint_bias.ModelToMotorKd(i, m.kd()),
            .tau = joint_bias.ModelToMotorTau(i, m.tau()),
        };
        write_motor_frame(can_id, EncodeType1StandardSerialFrame(args.channel, t1, ranges));
        ++stats.tx_type1_frames;
      }

      // --- 3) 快照 RX 线程更新的电机/IMU 缓存 ---
      std::array<MotorFeedbackCache, 12> cache_snapshot{};
      ImuCache imu_snapshot{};
      {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache_snapshot = motor_cache;
      }
      {
        std::lock_guard<std::mutex> lock(imu_mutex);
        imu_snapshot = imu_cache;
      }

      // --- 4) 在线 joint bias 标定（若启用且尚未完成）---
      if (!joint_bias.calibrated()) {
        std::array<float, 12> q_motor{};
        std::array<bool, 12> seen{};
        for (size_t i = 0; i < kJointOrder.size(); ++i) {
          q_motor[i] = cache_snapshot[i].q;
          seen[i] = cache_snapshot[i].seen;
        }
        joint_bias.MaybeLogCalibProgress(seen);
        joint_bias.AccumulateSample(q_motor, seen);
        joint_bias.TryFinishCalibration(args.joint_bias_calib_seconds, kMinJointBiasSamples,
                                        args.joint_bias_calib_timeout_seconds);
      }

      // --- 5) 发布 rt/lowstate（频率 = tick_hz）---
      if (lowstate_pub->trylock()) {
        for (size_t i = 0; i < kJointOrder.size(); ++i) {
          auto& ms = lowstate_pub->msg_.motor_state()[i];
          const auto& cached = cache_snapshot[i];
          ms.q() = joint_bias.MotorToModel(i, cached.q);
          ms.dq() = joint_bias.MotorToModelDq(i, cached.dq);
          ms.tau_est() = joint_bias.MotorToModelTau(i, cached.tau_est);
          ms.q_raw() = cached.q;
          ms.dq_raw() = cached.dq;
          ms.temperature() = cached.temperature;
          ms.lost() = cached.seen ? 0u : 1u;
        }
        for (size_t i = 0; i < imu_snapshot.quaternion.size(); ++i) {
          lowstate_pub->msg_.imu_state().quaternion()[i] = imu_snapshot.quaternion[i];
        }
        for (size_t i = 0; i < imu_snapshot.gyroscope.size(); ++i) {
          lowstate_pub->msg_.imu_state().gyroscope()[i] = imu_snapshot.gyroscope[i];
        }
        lowstate_pub->msg_.tick() = tick++;
        lowstate_pub->unlockAndPublish();
      }

      if (tick == 1) {
        std::cout << "[PHASE1] main loop running at " << args.tick_hz << " Hz\n";
      }
      if ((tick % static_cast<uint32_t>(std::max(1.0, args.tick_hz))) == 0) {
        std::cout << "[STAT] rx_frames=" << stats.rx_frames.load()
                  << " rx_a=" << stats.rx_a_frames.load()
                  << " rx_b=" << stats.rx_b_frames.load()
                  << " type2_frames=" << stats.type2_frames.load()
                  << " type2_a=" << stats.type2_a_frames.load()
                  << " type2_b=" << stats.type2_b_frames.load()
                  << " decode_errors=" << stats.decode_errors.load()
                  << " tx_type1=" << stats.tx_type1_frames.load()
                  << " tx_enable=" << stats.tx_enable_frames.load()
                  << " tx_disable=" << stats.tx_disable_frames.load()
                  << " tx_a=" << stats.tx_a_frames.load()
                  << " tx_b=" << stats.tx_b_frames.load()
                  << " imu_frames=" << stats.imu_frames.load()
                  << " imu_errors=" << stats.imu_errors.load()
                  << " tx_write_errors=" << stats.tx_write_errors.load() << "\n";
      }

      const auto elapsed = std::chrono::steady_clock::now() - t0;
      if (elapsed < dt) std::this_thread::sleep_for(dt - elapsed);
    }

    if (args.disable_on_exit) {
      std::cout << "[PHASE1] shutting down: disable all 12 motors\n";
      for (auto j : kJointOrder) {
        const auto can_id = JointCanId(j);
        write_motor_frame(can_id, BuildType34(kLingzuMotorDisableCode, can_id, static_cast<uint8_t>(args.master_id),
                                              args.channel));
        ++stats.tx_disable_frames;
      }
    }
    std::cout << "[PHASE1] gateway exit\n";
    g_running = false;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}

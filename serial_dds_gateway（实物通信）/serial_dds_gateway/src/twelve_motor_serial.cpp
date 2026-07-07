#include "lingzu_motor_protocol.hpp"
#include "lingzu_serial.hpp"
#include "motor_map.hpp"
#include "protocol_codec.hpp"
#include "serial_framer.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

using namespace serial_dds_gateway;

namespace {

std::atomic<bool> g_running{true};

void OnSignal(int) { g_running = false; }

struct MotorCmdSpec {
  double q{0.0};
  double dq{0.0};
  double kp{0.0};
  double kd{0.5};
  double tau{0.0};
};

struct MotorFeedbackCache {
  double q{0.0};
  double dq{0.0};
  double tau{0.0};
  double temp_c{0.0};
  bool seen{false};
};

struct Args {
  std::string port_a;
  std::string port_b;
  std::string commands_file;
  int baudrate = 2000000;
  uint8_t channel = 0x00;
  uint16_t master_id = 0x00FD;
  double q = 0.0;
  double dq = 0.0;
  double kp = 0.0;
  double kd = 0.5;
  double tau = 0.0;
  double rx_seconds = 3.0;
  double tx_hz = 50.0;
  bool send_enable = false;
  bool send_disable = false;
  bool clear_fault = false;
  bool disable_on_exit = false;
};

std::string Trim(std::string s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
  return s;
}

std::vector<std::string> SplitCsv(const std::string& line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string cell;
  while (std::getline(ss, cell, ',')) out.push_back(Trim(cell));
  return out;
}

size_t JointIndex(std::string_view joint) {
  for (size_t i = 0; i < kJointOrder.size(); ++i) {
    if (kJointOrder[i] == joint) return i;
  }
  throw std::runtime_error(std::string("unknown joint: ") + std::string(joint));
}

size_t MotorIndex(uint8_t motor_id) {
  const auto it = kCanIdToJoint.find(motor_id);
  if (it == kCanIdToJoint.end()) {
    throw std::runtime_error("unknown motor id: " + std::to_string(static_cast<int>(motor_id)));
  }
  return JointIndex(it->second);
}

void LoadCommandsFile(const std::string& path, std::array<MotorCmdSpec, 12>& cmds) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open commands file: " + path);

  std::string line;
  size_t line_no = 0;
  size_t loaded = 0;
  while (std::getline(in, line)) {
    ++line_no;
    line = Trim(line);
    if (line.empty() || line[0] == '#') continue;

    const auto cols = SplitCsv(line);
    if (cols.size() == 6 && (cols[0] == "joint" || cols[1] == "q")) continue;  // header

    size_t index = 0;
    size_t value_offset = 0;
    if (cols.size() == 6) {
      if (kJointToCanId.count(cols[0]) != 0) {
        index = JointIndex(cols[0]);
      } else {
        index = MotorIndex(static_cast<uint8_t>(std::stoi(cols[0], nullptr, 0)));
      }
      value_offset = 1;
    } else if (cols.size() == 5) {
      index = loaded;
      if (index >= cmds.size()) {
        throw std::runtime_error("commands file has more than 12 motor rows");
      }
    } else {
      throw std::runtime_error("line " + std::to_string(line_no) +
                               ": expected joint,q,dq,kp,kd,tau or q,dq,kp,kd,tau");
    }

    cmds[index].q = std::stod(cols[value_offset + 0]);
    cmds[index].dq = std::stod(cols[value_offset + 1]);
    cmds[index].kp = std::stod(cols[value_offset + 2]);
    cmds[index].kd = std::stod(cols[value_offset + 3]);
    cmds[index].tau = std::stod(cols[value_offset + 4]);
    ++loaded;
  }
  if (loaded == 0) throw std::runtime_error("commands file has no motor rows: " + path);
}

std::array<MotorCmdSpec, 12> BuildMotorCommands(const Args& args) {
  std::array<MotorCmdSpec, 12> cmds{};
  for (auto& cmd : cmds) {
    cmd.q = args.q;
    cmd.dq = args.dq;
    cmd.kp = args.kp;
    cmd.kd = args.kd;
    cmd.tau = args.tau;
  }
  if (!args.commands_file.empty()) LoadCommandsFile(args.commands_file, cmds);
  return cmds;
}

Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto need = [&](const char* name) {
      if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + name);
      return std::string(argv[++i]);
    };
    if (k == "--port-a") a.port_a = need("--port-a");
    else if (k == "--port-b") a.port_b = need("--port-b");
    else if (k == "--commands-file") a.commands_file = need("--commands-file");
    else if (k == "--baudrate") a.baudrate = std::stoi(need("--baudrate"));
    else if (k == "--channel") a.channel = static_cast<uint8_t>(std::stoi(need("--channel"), nullptr, 0));
    else if (k == "--master-id") a.master_id = static_cast<uint16_t>(std::stoi(need("--master-id"), nullptr, 0));
    else if (k == "--q") a.q = std::stod(need("--q"));
    else if (k == "--dq") a.dq = std::stod(need("--dq"));
    else if (k == "--kp") a.kp = std::stod(need("--kp"));
    else if (k == "--kd") a.kd = std::stod(need("--kd"));
    else if (k == "--tau") a.tau = std::stod(need("--tau"));
    else if (k == "--rx-seconds") a.rx_seconds = std::stod(need("--rx-seconds"));
    else if (k == "--tx-hz") a.tx_hz = std::stod(need("--tx-hz"));
    else if (k == "--send-enable") a.send_enable = true;
    else if (k == "--send-disable") a.send_disable = true;
    else if (k == "--clear-fault") a.clear_fault = true;
    else if (k == "--disable-on-exit") a.disable_on_exit = true;
    else if (k == "--help" || k == "-h") {
      std::cout
          << "Usage: twelve_motor_serial --port-a /dev/myttyCAN0 --port-b /dev/myttyCAN1\n"
             "  [--commands-file config/twelve_motor_commands.example.csv]\n"
             "  [--baudrate 2000000] [--channel 0x00] [--master-id 0x00FD]\n"
             "  [--send-enable] [--send-disable] [--clear-fault] [--disable-on-exit]\n"
             "  [--q 0] [--dq 0] [--kp 0] [--kd 0.5] [--tau 0]   # defaults for motors not listed in CSV\n"
             "  [--tx-hz 50] [--rx-seconds 3]\n"
             "CSV columns: joint,q,dq,kp,kd,tau  (joint name or decimal motor id)\n"
             "Port A: FR/RR motors (11,21,31,13,23,33). Port B: FL/RL (12,22,32,14,24,34).\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + k);
    }
  }
  if (a.port_a.empty() || a.port_b.empty()) {
    throw std::runtime_error("Use both --port-a and --port-b");
  }
  if (a.tx_hz <= 0.0) throw std::runtime_error("--tx-hz must be > 0");
  return a;
}

void RxLoop(SerialFramer& framer, const RangeSpec& ranges, const std::unordered_map<uint8_t, size_t>& canid_to_index,
            std::array<MotorFeedbackCache, 12>& cache, std::mutex& cache_mutex, std::atomic<uint64_t>& rx_frames,
            std::atomic<uint64_t>& type2_frames, std::atomic<uint64_t>& decode_errors) {
  while (g_running) {
    const auto frames = framer.ReadAvailableFrames();
    rx_frames.fetch_add(frames.size());
    for (const auto& f : frames) {
      if (f.data.size() < 8) {
        ++decode_errors;
        continue;
      }
      Type2Feedback fb;
      try {
        fb = DecodeType2SerialFrame(f, ranges);
      } catch (const std::exception&) {
        ++decode_errors;
        continue;
      }
      const auto it = canid_to_index.find(fb.motor_id);
      if (it == canid_to_index.end()) {
        ++decode_errors;
        continue;
      }
      {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto& c = cache[it->second];
        c.q = fb.q;
        c.dq = fb.dq;
        c.tau = fb.tau;
        c.temp_c = fb.temp_c;
        c.seen = true;
      }
      ++type2_frames;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

SerialFramer& FramerForBus(SerialFramer& a, SerialFramer& b, MotorSerialBus bus) {
  return bus == MotorSerialBus::A ? a : b;
}

void WriteToMotor(SerialFramer& framer_a, SerialFramer& framer_b, uint8_t can_id, const SerialFrame& frame) {
  FramerForBus(framer_a, framer_b, MotorSerialBusForCanId(can_id)).WriteFrame(frame);
}

void SendType34All(SerialFramer& framer_a, SerialFramer& framer_b, const Args& args, uint8_t mode_code,
                   uint8_t data0 = 0) {
  const uint8_t master = static_cast<uint8_t>(args.master_id);
  for (const auto joint : kJointOrder) {
    const auto can_id = kJointToCanId.at(joint);
    WriteToMotor(framer_a, framer_b, can_id,
                 BuildMotorModeFrame(args.channel, master, can_id, mode_code, data0));
  }
}

void SendType1All(SerialFramer& framer_a, SerialFramer& framer_b, const Args& args, const RangeSpec& ranges,
                  const std::array<MotorCmdSpec, 12>& motor_cmds) {
  for (size_t i = 0; i < kJointOrder.size(); ++i) {
    const auto can_id = kJointToCanId.at(kJointOrder[i]);
    const auto& spec = motor_cmds[i];
    Type1Command cmd{
        .motor_id = can_id, .q = spec.q, .dq = spec.dq, .kp = spec.kp, .kd = spec.kd, .tau = spec.tau};
    WriteToMotor(framer_a, framer_b, can_id, EncodeType1StandardSerialFrame(args.channel, cmd, ranges));
  }
}

void PrintCommandTable(const std::array<MotorCmdSpec, 12>& motor_cmds) {
  std::cout << "\n=== commanded type1 ===\n";
  std::cout << std::left << std::setw(18) << "joint" << std::setw(6) << "id" << std::setw(10) << "q"
            << std::setw(10) << "dq" << std::setw(8) << "kp" << std::setw(8) << "kd" << std::setw(8) << "tau"
            << "\n";
  for (size_t i = 0; i < kJointOrder.size(); ++i) {
    const auto can_id = kJointToCanId.at(kJointOrder[i]);
    const auto& c = motor_cmds[i];
    std::cout << std::left << std::setw(18) << kJointOrder[i] << std::setw(6) << static_cast<int>(can_id)
              << std::setw(10) << std::fixed << std::setprecision(3) << c.q << std::setw(10) << c.dq
              << std::setw(8) << c.kp << std::setw(8) << c.kd << std::setw(8) << c.tau << "\n";
  }
}

void PrintSummary(const std::array<MotorFeedbackCache, 12>& cache, uint64_t rx_frames, uint64_t type2_frames,
                  uint64_t decode_errors) {
  std::cout << "\n=== feedback type2 ===\n";
  std::cout << "rx_frames=" << rx_frames << " type2_frames=" << type2_frames << " decode_errors=" << decode_errors
            << "\n";
  std::cout << std::left << std::setw(18) << "joint" << std::setw(6) << "id" << std::setw(5) << "bus" << std::setw(6)
            << "seen" << std::setw(10) << "q" << std::setw(10) << "dq" << std::setw(10) << "tau" << std::setw(8)
            << "tempC"
            << "\n";
  for (size_t i = 0; i < kJointOrder.size(); ++i) {
    const auto can_id = kJointToCanId.at(kJointOrder[i]);
    const auto bus = MotorSerialBusForCanId(can_id);
    const auto& c = cache[i];
    std::cout << std::left << std::setw(18) << kJointOrder[i] << std::setw(6) << static_cast<int>(can_id)
              << std::setw(5) << (bus == MotorSerialBus::A ? "A" : "B") << std::setw(6) << (c.seen ? "yes" : "no")
              << std::setw(10) << std::fixed << std::setprecision(3) << c.q << std::setw(10) << c.dq << std::setw(10)
              << c.tau << std::setw(8) << c.temp_c << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, OnSignal);
  std::signal(SIGTERM, OnSignal);

  try {
    const auto args = ParseArgs(argc, argv);
    const auto motor_cmds = BuildMotorCommands(args);
    RangeSpec ranges;
    SerialFramer framer_a(args.port_a, args.baudrate);
    SerialFramer framer_b(args.port_b, args.baudrate);

    std::cout << "[INFO] port A (FR/RR)=" << args.port_a << " port B (FL/RL)=" << args.port_b << " @"
              << args.baudrate << "\n";
    if (!args.commands_file.empty()) {
      std::cout << "[INFO] per-motor commands from " << args.commands_file << "\n";
    } else {
      std::cout << "[INFO] same command for all motors (use --commands-file for per-joint values)\n";
    }
    PrintCommandTable(motor_cmds);
    std::cout << "[INFO] tx_hz=" << args.tx_hz << " rx_seconds=" << args.rx_seconds << "\n";

    std::unordered_map<uint8_t, size_t> canid_to_index;
    for (size_t i = 0; i < kJointOrder.size(); ++i) {
      canid_to_index[kJointToCanId.at(kJointOrder[i])] = i;
    }
    std::array<MotorFeedbackCache, 12> cache{};
    std::mutex cache_mutex;
    std::atomic<uint64_t> rx_frames{0};
    std::atomic<uint64_t> type2_frames{0};
    std::atomic<uint64_t> decode_errors{0};

    std::thread rx_a(RxLoop, std::ref(framer_a), std::cref(ranges), std::cref(canid_to_index), std::ref(cache),
                   std::ref(cache_mutex), std::ref(rx_frames), std::ref(type2_frames), std::ref(decode_errors));
    std::thread rx_b(RxLoop, std::ref(framer_b), std::cref(ranges), std::cref(canid_to_index), std::ref(cache),
                   std::ref(cache_mutex), std::ref(rx_frames), std::ref(type2_frames), std::ref(decode_errors));

    if (args.clear_fault) {
      SendType34All(framer_a, framer_b, args, kLingzuMotorDisableCode, 1);
      std::cout << "[TX] clear-fault x12\n";
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (args.send_enable) {
      SendType34All(framer_a, framer_b, args, kLingzuMotorEnableCode);
      std::cout << "[TX] enable x12\n";
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (args.send_disable) {
      SendType34All(framer_a, framer_b, args, kLingzuMotorDisableCode);
      std::cout << "[TX] disable x12\n";
    } else {
      const auto tx_period = std::chrono::duration<double>(1.0 / args.tx_hz);
      const auto t_end = std::chrono::steady_clock::now() + std::chrono::duration<double>(args.rx_seconds);
      uint64_t tx_count = 0;
      while (g_running && std::chrono::steady_clock::now() < t_end) {
        const auto tick_start = std::chrono::steady_clock::now();
        SendType1All(framer_a, framer_b, args, ranges, motor_cmds);
        ++tx_count;
        if (tx_count == 1 || (tx_count % static_cast<uint64_t>(args.tx_hz) == 0)) {
          std::cout << "[TX] type1 x12 (batch " << tx_count << ")\n";
        }
        const auto tick_end = std::chrono::steady_clock::now();
        const auto sleep_for = tx_period - (tick_end - tick_start);
        if (sleep_for.count() > 0) {
          std::this_thread::sleep_for(sleep_for);
        }
      }
      std::cout << "[TX] sent " << tx_count << " batches (" << (tx_count * 12) << " frames)\n";
    }

    g_running = false;
    rx_a.join();
    rx_b.join();

    PrintSummary(cache, rx_frames.load(), type2_frames.load(), decode_errors.load());

    if (args.disable_on_exit) {
      SendType34All(framer_a, framer_b, args, kLingzuMotorDisableCode);
      std::cout << "[TX] disable-on-exit x12\n";
    }

    const size_t seen_count = [&] {
      size_t n = 0;
      for (const auto& c : cache) {
        if (c.seen) ++n;
      }
      return n;
    }();
    if (!args.send_disable && seen_count < 12) {
      std::cerr << "[WARN] only " << seen_count << "/12 motors returned type2 feedback\n";
      return 2;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}

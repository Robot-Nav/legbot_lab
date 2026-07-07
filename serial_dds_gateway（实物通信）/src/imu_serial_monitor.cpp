#include "imu_framer.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

using namespace serial_dds_gateway;

namespace {

std::atomic<bool> g_running{true};

void OnSignal(int) { g_running = false; }

struct Args {
  std::string port;
  int baudrate = 921600;
  double duration_seconds = 0.0;
  bool degrees = false;
};

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto need = [&](const char* name) {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return std::string(argv[++i]);
    };
    if (key == "--port") {
      args.port = need("--port");
    } else if (key == "--baudrate") {
      args.baudrate = std::stoi(need("--baudrate"));
    } else if (key == "--duration") {
      args.duration_seconds = std::stod(need("--duration"));
    } else if (key == "--degrees") {
      args.degrees = true;
    } else if (key == "--help" || key == "-h") {
      std::cout << "Usage: imu_serial_monitor --port /dev/myttyIMU [--baudrate 921600] "
                   "[--duration seconds] [--degrees]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }
  if (args.port.empty()) {
    throw std::runtime_error("use --port /dev/myttyIMU");
  }
  return args;
}

double MaybeDegrees(double radians, bool degrees) {
  constexpr double kRadToDeg = 57.29577951308232;
  return degrees ? radians * kRadToDeg : radians;
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, OnSignal);
  std::signal(SIGTERM, OnSignal);

  try {
    const auto args = ParseArgs(argc, argv);
    ImuFramer framer(args.port, args.baudrate);

    std::cout << "[INFO] reading IMU from " << args.port << " @" << args.baudrate << "\n";
    std::cout << "[INFO] expected frame: EB 90 A5 FF + 6*float32 little-endian + CRC16 + 80 7F\n";
    if (args.duration_seconds > 0.0) {
      std::cout << "[INFO] duration=" << args.duration_seconds << " seconds\n";
    } else {
      std::cout << "[INFO] press Ctrl+C to stop\n";
    }

    const auto start = std::chrono::steady_clock::now();
    uint64_t sample_count = 0;
    while (g_running) {
      for (const auto& sample : framer.ReadAvailableSamples()) {
        const auto q = EulerYprToQuaternion(sample.yaw, sample.pitch, sample.roll);
        ++sample_count;
        const char* angle_unit = args.degrees ? "deg" : "rad";
        const char* gyro_unit = args.degrees ? "deg/s" : "rad/s";
        std::cout << std::fixed << std::setprecision(5) << "[IMU " << sample_count << "] "
                  << "yaw=" << MaybeDegrees(sample.yaw, args.degrees) << " " << angle_unit
                  << " pitch=" << MaybeDegrees(sample.pitch, args.degrees) << " " << angle_unit
                  << " roll=" << MaybeDegrees(sample.roll, args.degrees) << " " << angle_unit
                  << " gx=" << MaybeDegrees(sample.gx, args.degrees) << " " << gyro_unit
                  << " gy=" << MaybeDegrees(sample.gy, args.degrees) << " " << gyro_unit
                  << " gz=" << MaybeDegrees(sample.gz, args.degrees) << " " << gyro_unit
                  << " quat=[" << q.w << ", " << q.x << ", " << q.y << ", " << q.z << "]\n";
      }

      if (args.duration_seconds > 0.0) {
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (elapsed >= args.duration_seconds) {
          break;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    std::cout << "[INFO] received " << sample_count << " valid IMU frame(s)\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}

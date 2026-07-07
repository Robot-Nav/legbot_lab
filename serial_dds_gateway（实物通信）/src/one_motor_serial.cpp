#include "lingzu_serial.hpp"
#include "lingzu_motor_protocol.hpp"
#include "motor_map.hpp"
#include "protocol_codec.hpp"
#include "serial_framer.hpp"

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace serial_dds_gateway;

struct Args {
  std::string port;
  int baudrate = 2000000;
  uint8_t channel = 0x00;
  uint8_t motor_id = 0x01;
  uint16_t master_id = 0x00FD;
  double q = 1.0;
  double dq = 0.0;
  double kp = 30.0;
  double kd = 1.0;
  double tau = 0.0;
  double rx_seconds = 2.0;
  bool send_enable = false;
  bool send_disable = false;
  bool clear_fault = false;
};

Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto need = [&](const char* name) {
      if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + name);
      return std::string(argv[++i]);
    };
    if (k == "--port") a.port = need("--port");
    else if (k == "--baudrate") a.baudrate = std::stoi(need("--baudrate"));
    else if (k == "--channel") a.channel = static_cast<uint8_t>(std::stoi(need("--channel"), nullptr, 0));
    else if (k == "--motor-id") a.motor_id = static_cast<uint8_t>(std::stoi(need("--motor-id"), nullptr, 0));
    else if (k == "--master-id") a.master_id = static_cast<uint16_t>(std::stoi(need("--master-id"), nullptr, 0));
    else if (k == "--q") a.q = std::stod(need("--q"));
    else if (k == "--dq") a.dq = std::stod(need("--dq"));
    else if (k == "--kp") a.kp = std::stod(need("--kp"));
    else if (k == "--kd") a.kd = std::stod(need("--kd"));
    else if (k == "--tau") a.tau = std::stod(need("--tau"));
    else if (k == "--rx-seconds") a.rx_seconds = std::stod(need("--rx-seconds"));
    else if (k == "--send-enable") a.send_enable = true;
    else if (k == "--send-disable") a.send_disable = true;
    else if (k == "--clear-fault") a.clear_fault = true;
  }
  if (a.port.empty()) throw std::runtime_error("Use --port /dev/myttyCAN0 or /dev/myttyCAN1");
  return a;
}

int main(int argc, char** argv) {
  try {
    const auto args = ParseArgs(argc, argv);
    RangeSpec ranges;
    SerialFramer framer(args.port, args.baudrate);

    if (args.send_enable) {
      framer.WriteFrame(BuildMotorEnableFrame(args.channel, static_cast<uint8_t>(args.master_id), args.motor_id));
      std::cout << "[TX] enable motor_id=" << static_cast<int>(args.motor_id) << "\n";
    }
    if (args.clear_fault) {
      framer.WriteFrame(BuildMotorClearFaultFrame(args.channel, static_cast<uint8_t>(args.master_id), args.motor_id));
      std::cout << "[TX] clear-fault motor_id=" << static_cast<int>(args.motor_id) << "\n";
    }
    if (args.send_disable) {
      framer.WriteFrame(BuildMotorDisableFrame(args.channel, static_cast<uint8_t>(args.master_id), args.motor_id));
      std::cout << "[TX] disable motor_id=" << static_cast<int>(args.motor_id) << "\n";
    }

    Type1Command cmd{
        .motor_id = args.motor_id, .q = args.q, .dq = args.dq, .kp = args.kp, .kd = args.kd, .tau = args.tau};
    auto tx = EncodeType1StandardSerialFrame(args.channel, cmd, ranges);
    framer.WriteFrame(tx);
    std::cout << "[TX] channel=" << static_cast<int>(tx.channel) << " frame_type=0x" << std::hex << std::uppercase
              << static_cast<int>(tx.frame_type) << " id_field=0x" << tx.id_field << " master_id=0x"
              << static_cast<int>(tx.master_id) << std::dec << " motor_id=" << static_cast<int>(args.motor_id)
              << "\n";

    const auto t_end = std::chrono::steady_clock::now() + std::chrono::duration<double>(args.rx_seconds);
    while (std::chrono::steady_clock::now() < t_end) {
      for (const auto& f : framer.ReadAvailableFrames()) {
        std::cout << "[RX] channel=" << static_cast<int>(f.channel) << " frame_type=0x" << std::hex << std::uppercase
                  << static_cast<int>(f.frame_type) << " id_field=0x" << f.id_field << " master_id=0x"
                  << static_cast<int>(f.master_id) << std::dec << " data=";
        for (auto b : f.data) std::cout << " " << std::hex << std::uppercase << static_cast<int>(b);
        std::cout << std::dec << "\n";
        if (f.data.size() == 8) {
          try {
            auto fb = DecodeType2SerialFrame(f, ranges);
            std::cout << "     q=" << fb.q << " dq=" << fb.dq << " tau=" << fb.tau << " temp=" << fb.temp_c << "\n";
          } catch (const std::exception& e) {
            std::cout << "     skip feedback decode: " << e.what() << "\n";
          }
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}


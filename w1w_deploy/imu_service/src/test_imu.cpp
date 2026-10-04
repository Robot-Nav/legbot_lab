#include "dm_imu/imu_driver.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

namespace
{
std::atomic_bool g_keep_running{true};

void handle_signal(int)
{
  g_keep_running.store(false);
}

void print_usage(const char* exe)
{
  std::cout << "Usage: " << exe << " [serial_port] [baudrate]" << std::endl;
}
} // namespace

int main(int argc, char** argv)
{
  std::string port = "/dev/ttyACM0";
  int baud = 921600;

  if (argc > 1)
  {
    const std::string arg = argv[1];
    if (arg == "--help" || arg == "-h")
    {
      print_usage(argv[0]);
      return EXIT_SUCCESS;
    }
    port = arg;
  }
  if (argc > 2)
  {
    baud = std::atoi(argv[2]);
    if (baud <= 0)
    {
      std::cerr << "Invalid baudrate " << argv[2] << std::endl;
      return EXIT_FAILURE;
    }
  }

  auto callback = [](const dmbot_serial::IMU_Data& data)
  {
    std::cout << std::fixed << std::setprecision(4)
          << "acc[g] " << data.accx << ", " << data.accy << ", " << data.accz
          << " | gyro[rad/s] " << data.gyrox << ", " << data.gyroy << ", " << data.gyroz
          << " | rpy[rad] " << data.roll << ", " << data.pitch << ", " << data.yaw
          << '\r' << std::flush;
  };

  dmbot_serial::DmImu imu(port, baud, callback);

  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);

  while (g_keep_running.load())
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  imu.stop();
  std::cout << std::endl;
  return EXIT_SUCCESS;
}





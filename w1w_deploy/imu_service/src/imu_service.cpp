#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "dm_imu/imu_driver.h"
#include "w1w/device_guard.h"

namespace
{

constexpr uint32_t kCommandMagic = 0x494D5543; // "IMUC"
constexpr uint32_t kFeedbackMagic = 0x494D5546; // "IMUF"
constexpr uint16_t kProtocolVersion = 1;
constexpr uint32_t kFeedbackStatusDataStale = 0x1U;

std::atomic_bool g_keep_running{true};

void handle_signal(int)
{
    g_keep_running.store(false);
}

#pragma pack(push, 1)
struct ImuCommandNet
{
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    uint32_t sequence;
    uint32_t reserved;
};

struct ImuFeedbackNet
{
    uint32_t magic;
    uint16_t version;
    uint16_t reserved;
    uint32_t sequence;
    uint32_t data_sequence;
    uint32_t command_sequence;
    uint32_t status_bits;
    float acceleration[3];
    float angular_velocity[3];
    float rpy[3];
    float data_age_ms;
    float sample_rate_hz;
    float tx_rate_hz;
};
#pragma pack(pop)

static_assert(sizeof(ImuCommandNet) == 16, "Unexpected IMU command packet layout");
static_assert(sizeof(ImuFeedbackNet) == 72, "Unexpected IMU feedback packet layout");

struct SharedState
{
    std::mutex mutex;
    dmbot_serial::IMU_Data latest{};
    std::chrono::steady_clock::time_point timestamp{};
    uint32_t sequence = 0;
    std::uint64_t total_samples = 0;
    bool has_data = false;
};

struct ClientPeer
{
    sockaddr_in address{};
    std::chrono::steady_clock::time_point last_seen{};
    uint32_t command_sequence = 0;
};

bool same_client(const sockaddr_in& left, const sockaddr_in& right)
{
    return left.sin_family == right.sin_family &&
           left.sin_addr.s_addr == right.sin_addr.s_addr &&
           left.sin_port == right.sin_port;
}

void remember_peer(std::vector<ClientPeer>& peers,
                   const sockaddr_in& address,
                   uint32_t command_sequence,
                   std::chrono::steady_clock::time_point now)
{
    for (auto& peer : peers)
    {
        if (same_client(peer.address, address))
        {
            peer.last_seen = now;
            peer.command_sequence = command_sequence;
            return;
        }
    }
    if (peers.size() >= 8)
    {
        auto oldest = std::min_element(peers.begin(), peers.end(),
            [](const ClientPeer& left, const ClientPeer& right) {
                return left.last_seen < right.last_seen;
            });
        *oldest = ClientPeer{address, now, command_sequence};
        return;
    }
    peers.push_back(ClientPeer{address, now, command_sequence});
}

bool set_non_blocking(int fd)
{
    const int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0)
    {
        return false;
    }
    if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0)
    {
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!w1w::device_guard::enforce("imu_service"))
    {
        return 77;
    }

    std::string device = "/dev/ttyACM0";
    std::string bind_host = "127.0.0.1";
    int baudrate = 921600;
    uint16_t udp_port = 55200;
    int cycle_us = 1000;
    std::chrono::milliseconds data_timeout{100};

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc)
        {
            device = argv[++i];
        }
        else if (arg == "--bind-host" && i + 1 < argc)
        {
            bind_host = argv[++i];
        }
        else if (arg == "--baud" && i + 1 < argc)
        {
            baudrate = std::stoi(argv[++i]);
        }
        else if (arg == "--port" && i + 1 < argc)
        {
            udp_port = static_cast<uint16_t>(std::stoi(argv[++i]));
        }
        else if (arg == "--cycle-us" && i + 1 < argc)
        {
            cycle_us = std::stoi(argv[++i]);
        }
        else if (arg == "--timeout-ms" && i + 1 < argc)
        {
            data_timeout = std::chrono::milliseconds(std::stoi(argv[++i]));
        }
        else if (arg == "--help")
        {
            std::cout << "Usage: " << argv[0]
                      << " [--device <serial port>] [--baud <baudrate>]"
                      << " [--bind-host <IPv4 address>] [--port <udp port>]"
                      << " [--cycle-us <microseconds>] [--timeout-ms <ms>]"
                      << std::endl;
            return 0;
        }
    }

    if (cycle_us <= 0)
    {
        std::cerr << "cycle-us must be > 0" << std::endl;
        return 1;
    }
    if (baudrate <= 0)
    {
        std::cerr << "baud must be > 0" << std::endl;
        return 1;
    }

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    try
    {
        SharedState shared;

        auto imu_callback = [&shared](const dmbot_serial::IMU_Data& data)
        {
            std::lock_guard<std::mutex> lock(shared.mutex);
            shared.latest = data;
            shared.timestamp = std::chrono::steady_clock::now();
            ++shared.sequence;
            ++shared.total_samples;
            shared.has_data = true;
        };

        dmbot_serial::DmImu imu(device, baudrate, imu_callback);

        const int udp_socket = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (udp_socket < 0)
        {
            throw std::runtime_error("Failed to open UDP socket");
        }

        sockaddr_in bind_addr{};
        bind_addr.sin_family = AF_INET;
        if (inet_pton(AF_INET, bind_host.c_str(), &bind_addr.sin_addr) != 1)
        {
            ::close(udp_socket);
            throw std::runtime_error("Invalid IPv4 bind address: " + bind_host);
        }
        bind_addr.sin_port = htons(udp_port);

        int reuse = 1;
        setsockopt(udp_socket, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

        if (bind(udp_socket, reinterpret_cast<sockaddr*>(&bind_addr), sizeof(bind_addr)) < 0)
        {
            ::close(udp_socket);
            throw std::runtime_error("Failed to bind UDP port");
        }

        if (!set_non_blocking(udp_socket))
        {
            ::close(udp_socket);
            throw std::runtime_error("Failed to set UDP socket non-blocking");
        }

        std::vector<ClientPeer> feedback_peers;
        uint32_t feedback_sequence = 0;
        std::uint64_t send_count = 0;
        std::uint64_t send_count_last = 0;
        std::uint64_t sample_count_last = 0;
        float sample_rate_value = 0.0F;
        float tx_rate_value = 0.0F;
        auto last_rate_update = std::chrono::steady_clock::now();

        const auto cycle_period = std::chrono::microseconds(cycle_us);
        auto next_tick = std::chrono::steady_clock::now() + cycle_period;

        while (g_keep_running.load())
        {
            while (true)
            {
                ImuCommandNet packet{};
                sockaddr_in from_addr{};
                socklen_t from_len = sizeof(from_addr);
                const ssize_t bytes = recvfrom(udp_socket, &packet, sizeof(packet), 0,
                                               reinterpret_cast<sockaddr*>(&from_addr), &from_len);
                if (bytes < 0)
                {
                    if (errno == EAGAIN || errno == EWOULDBLOCK)
                    {
                        break;
                    }
                    continue;
                }
                if (static_cast<std::size_t>(bytes) != sizeof(ImuCommandNet))
                {
                    continue;
                }
                if (packet.magic != kCommandMagic ||
                    packet.version != kProtocolVersion ||
                    packet.flags != 0U ||
                    packet.reserved != 0U)
                {
                    continue;
                }

                remember_peer(feedback_peers, from_addr, packet.sequence,
                              std::chrono::steady_clock::now());
            }
            auto now = std::chrono::steady_clock::now();

            const auto peer_expiry = now - std::chrono::seconds(2);
            feedback_peers.erase(
                std::remove_if(feedback_peers.begin(), feedback_peers.end(),
                    [peer_expiry](const ClientPeer& peer) {
                        return peer.last_seen < peer_expiry;
                    }),
                feedback_peers.end());

            dmbot_serial::IMU_Data sample{};
            uint32_t sample_sequence = 0;
            std::uint64_t sample_count = 0;
            std::chrono::steady_clock::time_point sample_time{};
            bool have_data = false;
            {
                std::lock_guard<std::mutex> lock(shared.mutex);
                sample = shared.latest;
                sample_sequence = shared.sequence;
                sample_time = shared.timestamp;
                sample_count = shared.total_samples;
                have_data = shared.has_data;
            }

            const float data_age_ms = have_data
                                          ? static_cast<float>(std::chrono::duration<double, std::milli>(now - sample_time).count())
                                          : static_cast<float>(data_timeout.count());
            const bool stale_data = !have_data || data_age_ms > static_cast<float>(data_timeout.count());

            if (now - last_rate_update >= std::chrono::seconds(1))
            {
                const double period = std::chrono::duration<double>(now - last_rate_update).count();
                sample_rate_value = static_cast<float>((sample_count - sample_count_last) / period);
                tx_rate_value = static_cast<float>((send_count - send_count_last) / period);
                sample_count_last = sample_count;
                send_count_last = send_count;
                last_rate_update = now;
            }

            for (const ClientPeer& peer : feedback_peers)
            {
                ImuFeedbackNet feedback{};
                feedback.magic = kFeedbackMagic;
                feedback.version = kProtocolVersion;
                feedback.sequence = feedback_sequence;
                feedback.data_sequence = sample_sequence;
                feedback.command_sequence = peer.command_sequence;
                feedback.status_bits = stale_data ? kFeedbackStatusDataStale : 0U;
                feedback.acceleration[0] = sample.accx;
                feedback.acceleration[1] = sample.accy;
                feedback.acceleration[2] = sample.accz;
                feedback.angular_velocity[0] = sample.gyrox;
                feedback.angular_velocity[1] = sample.gyroy;
                feedback.angular_velocity[2] = sample.gyroz;
                feedback.rpy[0] = sample.roll;
                feedback.rpy[1] = sample.pitch;
                feedback.rpy[2] = sample.yaw;
                feedback.data_age_ms = data_age_ms;
                feedback.sample_rate_hz = sample_rate_value;
                feedback.tx_rate_hz = tx_rate_value;

                const ssize_t sent = sendto(udp_socket, &feedback, sizeof(feedback), 0,
                                            reinterpret_cast<const sockaddr*>(&peer.address),
                                            sizeof(peer.address));
                if (sent >= 0)
                {
                    ++send_count;
                }
            }
            ++feedback_sequence;

            std::this_thread::sleep_until(next_tick);
            next_tick += cycle_period;
        }

        ::close(udp_socket);
        imu.stop();
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}

#include "elrs/elrs_driver.h"
#include "w1w/device_guard.h"

#include <algorithm>
#include <arpa/inet.h>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <iostream>
#include <mutex>
#include <netinet/in.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace
{
constexpr uint32_t kCommandMagic = 0x454C5243;
constexpr uint32_t kFeedbackMagic = 0x454C5246;
constexpr uint16_t kVersion = 1;
constexpr uint16_t kCommandVersion = 2;
constexpr uint32_t kStatusStale = 1U;
constexpr uint16_t kCommandFlagMotorTemperatures = 1U;
constexpr uint8_t kCrsfBatteryFrame = 0x08;
constexpr uint8_t kCrsfTemperatureFrame = 0x0D;
constexpr uint8_t kCrsfFlightModeFrame = 0x21;
constexpr uint8_t kTemperatureSensorId = 1;
constexpr std::size_t kMotorCount = 16;
constexpr std::array<uint8_t, 11> kMaxTemperatureLabel{
    'M', 'A', 'X', ' ', 'T', 'E', 'M', 'P', ' ', 'C', 0,
};
std::atomic_bool g_running{true};

void stop_handler(int)
{
    g_running.store(false);
}

int64_t monotonic_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

#pragma pack(push, 1)
struct CommandPacket
{
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    uint32_t sequence;
    uint32_t reserved;
};

struct CommandPacketV2
{
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    uint32_t sequence;
    uint32_t reserved;
    uint16_t motor_temperatures_deci_c[kMotorCount];
};

struct FeedbackPacket
{
    uint32_t magic;
    uint16_t version;
    uint16_t reserved;
    uint32_t sequence;
    uint32_t data_sequence;
    uint32_t command_sequence;
    uint32_t status_bits;
    uint16_t channels[16];
    uint8_t frame_id;
    uint8_t padding;
    float data_age_ms;
    float sample_rate_hz;
    float tx_rate_hz;
};
#pragma pack(pop)

static_assert(sizeof(CommandPacket) == 16, "ELRS command packet size changed");
static_assert(sizeof(CommandPacketV2) == 48, "ELRS v2 command packet size changed");
static_assert(sizeof(FeedbackPacket) == 70, "ELRS feedback packet size changed");

struct SharedState
{
    std::mutex mutex;
    elrs_serial::ELRS_Data data{};
    int64_t timestamp_ns{};
    uint32_t sequence{};
    uint64_t samples{};
};

struct ClientState
{
    sockaddr_in address{};
    uint32_t command_sequence{};
    uint32_t feedback_sequence{};
    std::chrono::steady_clock::time_point last_seen{};
};

bool same_endpoint(const sockaddr_in& left, const sockaddr_in& right)
{
    return left.sin_addr.s_addr == right.sin_addr.s_addr && left.sin_port == right.sin_port;
}
} // namespace

int main(int argc, char** argv)
{
    if (!w1w::device_guard::enforce("elrs_service"))
    {
        return 77;
    }

    std::string device = "/dev/ttyS6";
    int baud = 420000;
    int port = 55201;
    int cycle_us = 1000;
    int timeout_ms = 100;
    bool stats = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) device = argv[++i];
        else if (arg == "--baud" && i + 1 < argc) baud = std::stoi(argv[++i]);
        else if (arg == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
        else if (arg == "--cycle-us" && i + 1 < argc) cycle_us = std::stoi(argv[++i]);
        else if (arg == "--timeout-ms" && i + 1 < argc) timeout_ms = std::stoi(argv[++i]);
        else if (arg == "--stats") stats = true;
        else if (arg == "--help")
        {
            std::cout << "Usage: " << argv[0]
                      << " [--device PATH] [--baud RATE] [--port PORT]"
                         " [--cycle-us US] [--timeout-ms MS] [--stats]\n";
            return 0;
        }
        else
        {
            std::cerr << "invalid or incomplete argument: " << arg << '\n';
            return 2;
        }
    }
    if (baud <= 0 || port <= 0 || port > 65535 || cycle_us <= 0 || timeout_ms <= 0)
    {
        std::cerr << "ELRS numeric arguments must be positive and port must fit uint16\n";
        return 2;
    }

    std::signal(SIGINT, stop_handler);
    std::signal(SIGTERM, stop_handler);
    try
    {
        SharedState shared;
        elrs_serial::ElrsReceiver receiver(device, baud, [&shared](const auto& data) {
            std::lock_guard<std::mutex> lock(shared.mutex);
            shared.data = data;
            shared.timestamp_ns = monotonic_ns();
            ++shared.sequence;
            ++shared.samples;
        });

        const int sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock < 0) throw std::runtime_error("failed to create ELRS UDP socket");
        sockaddr_in bind_addr{};
        bind_addr.sin_family = AF_INET;
        bind_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        bind_addr.sin_port = htons(static_cast<uint16_t>(port));
        if (bind(sock, reinterpret_cast<sockaddr*>(&bind_addr), sizeof(bind_addr)) != 0)
        {
            close(sock);
            throw std::runtime_error("failed to bind ELRS loopback UDP port");
        }
        const int flags = fcntl(sock, F_GETFL, 0);
        if (flags < 0 || fcntl(sock, F_SETFL, flags | O_NONBLOCK) != 0)
        {
            close(sock);
            throw std::runtime_error("failed to make ELRS UDP socket nonblocking");
        }

        std::vector<ClientState> clients;
        uint64_t sent = 0, last_sent = 0, last_samples = 0;
        std::array<uint16_t, kMotorCount> motor_temperatures{};
        bool motor_temperatures_valid = false;
        auto motor_temperatures_time = std::chrono::steady_clock::time_point{};
        uint64_t telemetry_sent = 0;
        uint32_t telemetry_slot = 0;
        float sample_hz = 0.0F, tx_hz = 0.0F;
        auto rate_time = std::chrono::steady_clock::now();
        const auto cycle = std::chrono::microseconds(cycle_us);
        auto next = std::chrono::steady_clock::now() + cycle;
        auto next_telemetry = std::chrono::steady_clock::now();

        std::cout << "ELRS service: " << device << " @ " << baud
                  << ", UDP 127.0.0.1:" << port << '\n';
        while (g_running.load())
        {
            while (true)
            {
                std::array<uint8_t, sizeof(CommandPacketV2)> command_bytes{};
                sockaddr_in source{};
                socklen_t source_len = sizeof(source);
                const ssize_t count = recvfrom(sock, command_bytes.data(), command_bytes.size(), 0,
                    reinterpret_cast<sockaddr*>(&source), &source_len);
                if (count < 0)
                {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) break;
                    continue;
                }
                CommandPacket command{};
                bool valid_command = false;
                if (count == static_cast<ssize_t>(sizeof(CommandPacket)))
                {
                    std::memcpy(&command, command_bytes.data(), sizeof(command));
                    valid_command = command.magic == kCommandMagic && command.version == kVersion;
                }
                else if (count == static_cast<ssize_t>(sizeof(CommandPacketV2)))
                {
                    CommandPacketV2 command_v2{};
                    std::memcpy(&command_v2, command_bytes.data(), sizeof(command_v2));
                    valid_command = command_v2.magic == kCommandMagic &&
                                    command_v2.version == kCommandVersion;
                    if (valid_command)
                    {
                        command.magic = command_v2.magic;
                        command.version = command_v2.version;
                        command.flags = command_v2.flags;
                        command.sequence = command_v2.sequence;
                        command.reserved = command_v2.reserved;
                        if ((command_v2.flags & kCommandFlagMotorTemperatures) != 0)
                        {
                            std::copy_n(command_v2.motor_temperatures_deci_c,
                                        kMotorCount, motor_temperatures.begin());
                            motor_temperatures_valid = true;
                            motor_temperatures_time = std::chrono::steady_clock::now();
                        }
                    }
                }
                if (valid_command && source.sin_addr.s_addr == htonl(INADDR_LOOPBACK))
                {
                    const auto found = std::find_if(clients.begin(), clients.end(),
                        [&source](const ClientState& item) {
                            return same_endpoint(item.address, source);
                        });
                    if (found == clients.end())
                    {
                        if (clients.size() >= 8) clients.erase(clients.begin());
                        clients.push_back(ClientState{
                            source, command.sequence, 0, std::chrono::steady_clock::now()});
                    }
                    else
                    {
                        found->command_sequence = command.sequence;
                        found->last_seen = std::chrono::steady_clock::now();
                    }
                }
            }

            elrs_serial::ELRS_Data current{};
            int64_t timestamp = 0;
            uint32_t data_sequence = 0;
            uint64_t samples = 0;
            {
                std::lock_guard<std::mutex> lock(shared.mutex);
                current = shared.data;
                timestamp = shared.timestamp_ns;
                data_sequence = shared.sequence;
                samples = shared.samples;
            }
            const int64_t now_ns = monotonic_ns();
            const float age_ms = samples == 0
                                     ? 1.0e9F
                                     : static_cast<float>((now_ns - timestamp) / 1.0e6);
            const bool stale = samples == 0 || age_ms > static_cast<float>(timeout_ms);

            const auto now = std::chrono::steady_clock::now();
            if (now >= next_telemetry)
            {
                if (motor_temperatures_valid &&
                    now - motor_temperatures_time < std::chrono::seconds(1))
                {
                    std::size_t max_index = 0;
                    for (std::size_t i = 1; i < kMotorCount; ++i)
                    {
                        if (motor_temperatures[i] > motor_temperatures[max_index])
                        {
                            max_index = i;
                        }
                    }
                    const uint32_t slot = telemetry_slot++ % 10U;
                    if (slot == 8U)
                    {
                        std::array<uint8_t, 1 + kMotorCount * 2> payload{};
                        payload[0] = kTemperatureSensorId;
                        for (std::size_t i = 0; i < kMotorCount; ++i)
                        {
                            payload[1 + i * 2] =
                                static_cast<uint8_t>(motor_temperatures[i] >> 8U);
                            payload[2 + i * 2] =
                                static_cast<uint8_t>(motor_temperatures[i] & 0xFFU);
                        }
                        if (receiver.write_crsf_frame(
                                kCrsfTemperatureFrame, payload.data(), payload.size()))
                        {
                            ++telemetry_sent;
                        }
                    }
                    else if (slot == 9U)
                    {
                        if (receiver.write_crsf_frame(
                                kCrsfFlightModeFrame,
                                kMaxTemperatureLabel.data(),
                                kMaxTemperatureLabel.size()))
                        {
                            ++telemetry_sent;
                        }
                    }
                    else
                    {
                        // AX12's current ELRS firmware does not forward TEMP
                        // frames, but it reliably forwards BATTERY. Publish
                        // max temperature in the voltage field (0.1 units)
                        // and mark every other battery field unavailable.
                        const std::array<uint8_t, 8> max_payload{
                            static_cast<uint8_t>(motor_temperatures[max_index] >> 8U),
                            static_cast<uint8_t>(motor_temperatures[max_index] & 0xFFU),
                            0xFF, 0xFF,
                            0xFF, 0xFF, 0xFF,
                            0xFF,
                        };
                        if (receiver.write_crsf_frame(
                                kCrsfBatteryFrame,
                                max_payload.data(), max_payload.size()))
                        {
                            ++telemetry_sent;
                        }
                    }
                }
                next_telemetry = now + std::chrono::milliseconds(200);
            }
            if (now - rate_time >= std::chrono::seconds(1))
            {
                const double seconds = std::chrono::duration<double>(now - rate_time).count();
                sample_hz = static_cast<float>((samples - last_samples) / seconds);
                tx_hz = static_cast<float>((sent - last_sent) / seconds);
                last_samples = samples;
                last_sent = sent;
                rate_time = now;
                if (stats)
                {
                    std::cout << "ELRS sample=" << sample_hz << "Hz tx=" << tx_hz
                              << "Hz age=" << age_ms << "ms seq=" << data_sequence
                              << " temp_tx=" << telemetry_sent << '\n';
                }
            }

            clients.erase(
                std::remove_if(clients.begin(), clients.end(), [&now](const ClientState& client) {
                    return now - client.last_seen > std::chrono::seconds(1);
                }),
                clients.end());
            for (ClientState& client : clients)
            {
                FeedbackPacket feedback{};
                feedback.magic = kFeedbackMagic;
                feedback.version = kVersion;
                feedback.sequence = client.feedback_sequence;
                feedback.data_sequence = data_sequence;
                feedback.command_sequence = client.command_sequence;
                feedback.status_bits = stale ? kStatusStale : 0U;
                std::memcpy(feedback.channels, current.channels, sizeof(feedback.channels));
                feedback.frame_id = current.frame_id;
                feedback.data_age_ms = age_ms;
                feedback.sample_rate_hz = sample_hz;
                feedback.tx_rate_hz = tx_hz;
                if (sendto(sock, &feedback, sizeof(feedback), 0,
                           reinterpret_cast<const sockaddr*>(&client.address),
                           sizeof(client.address)) >= 0)
                {
                    ++sent;
                    ++client.feedback_sequence;
                }
            }
            std::this_thread::sleep_until(next);
            next += cycle;
        }
        close(sock);
        receiver.stop();
    }
    catch (const std::exception& error)
    {
        std::cerr << "ELRS service error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

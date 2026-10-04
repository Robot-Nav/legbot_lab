#ifndef W1W_ELRS_DRIVER_H
#define W1W_ELRS_DRIVER_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace elrs_serial
{

struct ELRS_Data
{
    uint16_t channels[16]{};
    uint8_t frame_id{};
};

bool parse_crsf_rc_frame(const uint8_t* frame, std::size_t len, ELRS_Data& data);
std::vector<uint8_t> build_crsf_frame(
    uint8_t frame_type, const uint8_t* payload, std::size_t payload_size);

class ElrsReceiver
{
public:
    using Callback = std::function<void(const ELRS_Data&)>;

    ElrsReceiver(const std::string& device, int baudrate, Callback callback);
    ~ElrsReceiver();
    void stop();
    bool write_crsf_frame(
        uint8_t frame_type, const uint8_t* payload, std::size_t payload_size);

private:
    void read_thread_func();

    std::string device_;
    int baudrate_;
    Callback callback_;
    int fd_;
    std::atomic_bool running_;
    std::thread read_thread_;
    std::mutex write_mutex_;
};

} // namespace elrs_serial

#endif

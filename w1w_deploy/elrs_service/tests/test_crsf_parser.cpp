#include "elrs/elrs_driver.h"

#include <array>
#include <cstdint>
#include <iostream>

int main()
{
    constexpr std::array<uint8_t, 26> frame{
        0xC8, 0x18, 0x16, 0xDF, 0x03, 0x1F, 0x35, 0xC1, 0x07,
        0xF0, 0xF2, 0x95, 0x0F, 0xE0, 0x00, 0x2F, 0x1F, 0xC0,
        0xF7, 0x09, 0x00, 0x00, 0x4C, 0x7C, 0xE2, 0x47,
    };
    constexpr std::array<uint16_t, 16> expected{
        991, 992, 1236, 992, 1792, 997, 997, 1792,
        1792, 997, 1792, 1275, 0, 0, 1811, 1811,
    };
    elrs_serial::ELRS_Data data{};
    if (!elrs_serial::parse_crsf_rc_frame(frame.data(), frame.size(), data))
    {
        std::cerr << "captured CRSF frame did not parse\n";
        return 1;
    }
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        if (data.channels[i] != expected[i])
        {
            std::cerr << "channel " << i + 1 << " mismatch\n";
            return 1;
        }
    }
    auto invalid = frame;
    invalid.back() ^= 1;
    if (elrs_serial::parse_crsf_rc_frame(invalid.data(), invalid.size(), data))
    {
        std::cerr << "bad CRC was accepted\n";
        return 1;
    }
    constexpr std::array<uint8_t, 3> temperature_payload{1, 0x01, 0x2C};
    const auto temperature_frame = elrs_serial::build_crsf_frame(
        0x0D, temperature_payload.data(), temperature_payload.size());
    if (temperature_frame.size() != temperature_payload.size() + 4 ||
        temperature_frame[0] != 0xC8 || temperature_frame[1] != 5 ||
        temperature_frame[2] != 0x0D)
    {
        std::cerr << "temperature CRSF frame was not encoded\n";
        return 1;
    }
    std::cout << "CRSF parser tests passed\n";
    return 0;
}

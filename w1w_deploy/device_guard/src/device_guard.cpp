#include "w1w/device_guard.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

#include <openssl/crypto.h>
#include <openssl/evp.h>

#ifndef W1W_AUTHORIZED_CPU_DIGEST_HEX
#error "W1W_AUTHORIZED_CPU_DIGEST_HEX must be supplied by the sealed build"
#endif

namespace w1w::device_guard {
namespace {

constexpr std::string_view kDigestHex = W1W_AUTHORIZED_CPU_DIGEST_HEX;
constexpr std::string_view kDomain = "w1w-device-v1";

std::string trim(std::string value)
{
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }).base();
    if (first >= last)
    {
        return {};
    }
    return std::string(first, last);
}

std::optional<std::string> read_cpu_serial()
{
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line))
    {
        const auto colon = line.find(':');
        if (colon == std::string::npos || trim(line.substr(0, colon)) != "Serial")
        {
            continue;
        }
        std::string serial = trim(line.substr(colon + 1));
        std::transform(serial.begin(), serial.end(), serial.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        const bool valid = serial.size() == 16 &&
            std::all_of(serial.begin(), serial.end(), [](unsigned char ch) {
                return std::isxdigit(ch) != 0;
            }) && serial != "0000000000000000";
        if (valid)
        {
            return serial;
        }
        return std::nullopt;
    }
    return std::nullopt;
}

int hex_value(char value)
{
    if (value >= '0' && value <= '9')
    {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f')
    {
        return value - 'a' + 10;
    }
    if (value >= 'A' && value <= 'F')
    {
        return value - 'A' + 10;
    }
    return -1;
}

std::optional<std::array<unsigned char, 32>> parse_expected_digest()
{
    if (kDigestHex.size() != 64)
    {
        return std::nullopt;
    }
    std::array<unsigned char, 32> digest{};
    for (std::size_t index = 0; index < digest.size(); ++index)
    {
        const int high = hex_value(kDigestHex[index * 2]);
        const int low = hex_value(kDigestHex[index * 2 + 1]);
        if (high < 0 || low < 0)
        {
            return std::nullopt;
        }
        digest[index] = static_cast<unsigned char>((high << 4) | low);
    }
    return digest;
}

std::optional<std::array<unsigned char, 32>> hash_serial(const std::string& serial)
{
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (context == nullptr)
    {
        return std::nullopt;
    }
    std::array<unsigned char, 32> digest{};
    unsigned int digest_size = 0;
    const unsigned char separator = 0;
    const bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
        EVP_DigestUpdate(context, kDomain.data(), kDomain.size()) == 1 &&
        EVP_DigestUpdate(context, &separator, sizeof(separator)) == 1 &&
        EVP_DigestUpdate(context, serial.data(), serial.size()) == 1 &&
        EVP_DigestFinal_ex(context, digest.data(), &digest_size) == 1;
    EVP_MD_CTX_free(context);
    if (!ok || digest_size != digest.size())
    {
        return std::nullopt;
    }
    return digest;
}

} // namespace

bool enforce(const char* component_name)
{
    const auto serial = read_cpu_serial();
    const auto expected = parse_expected_digest();
    const auto actual = serial ? hash_serial(*serial) : std::nullopt;
    const bool authorized = expected && actual &&
        CRYPTO_memcmp(expected->data(), actual->data(), expected->size()) == 0;
    if (!authorized)
    {
        std::cerr << (component_name != nullptr ? component_name : "w1w")
                  << ": unauthorized device" << std::endl;
    }
    return authorized;
}

} // namespace w1w::device_guard

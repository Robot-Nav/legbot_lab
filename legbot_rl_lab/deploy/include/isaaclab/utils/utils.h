// 文件用途：通用工具函数。提供偏航四元数提取与 CSV 文件加载。
#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <eigen3/Eigen/Dense>
#include <spdlog/spdlog.h>

namespace isaaclab
{

// 从机体姿态四元数中提取仅包含偏航（yaw）的四元数。
inline Eigen::Quaternionf yawQuaternion(const Eigen::Quaternionf& q) {
  float yaw = std::atan2(2.0f * (q.w() * q.z() + q.x() * q.y()), 1.0f - 2.0f * (q.y() * q.y() + q.z() * q.z()));
  float half_yaw = yaw * 0.5f;
  Eigen::Quaternionf ret(std::cos(half_yaw), 0.0f, 0.0f, std::sin(half_yaw));
  return ret.normalized();
};

// 按逗号分隔读取 CSV 文件，返回浮点矩阵。
inline std::vector<std::vector<float>> load_csv(const std::string& filename)
{
    std::vector<std::vector<float>> data;
    std::ifstream file(filename);
    if (!file.is_open())
    {
        spdlog::error("Error opening file: {}", filename);
        return data;
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ','))
        {
            try
            {
                row.push_back(std::stof(value));
            }
            catch (const std::invalid_argument& e)
            {
                spdlog::error("Invalid value in file: {}", value);
            }
        }
        data.push_back(row);
    }
    file.close();
    return data;
}

}
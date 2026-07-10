#pragma once

// 仿真参数配置头文件
// 负责从 YAML 配置文件和命令行读取仿真、DDS 通信、手柄及弹性绳参数。

#include <iostream>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>

namespace param
{

// 仿真全局配置结构体
inline struct SimulationConfig
{
    std::filesystem::path robot_scene;  // 机器人场景 XML 文件路径

    int domain_id;                      // DDS 域 ID，隔离不同仿真实例
    std::string interface;              // DDS 网络接口名称

    int use_joystick;                   // 是否启用手柄或键盘输入
    std::string joystick_type;          // 手柄类型：xbox / switch / keyboard
    std::string joystick_device;        // 手柄设备节点路径
    int joystick_bits;                  // 摇杆量化位数，用于归一化到 [-1, 1]

    int print_scene_information;        // 是否打印模型结构信息，便于调试传感器索引

    int enable_elastic_band;            // 是否启用弹性绳约束
    int band_attached_link = 0;         // 弹性绳在 base 上的作用起始索引，等于 6 * body_id

    // 从 YAML 加载配置，字段缺失时直接退出，避免后续运行出现未定义行为
    void load_from_yaml(const std::string &filename)
    {
        auto cfg = YAML::LoadFile(filename);
        try
        {
            robot_scene = cfg["robot_scene"].as<std::string>();
            domain_id = cfg["domain_id"].as<int>();
            interface = cfg["interface"].as<std::string>();
            use_joystick = cfg["use_joystick"].as<int>();
            joystick_type = cfg["joystick_type"].as<std::string>();
            joystick_device = cfg["joystick_device"].as<std::string>();
            joystick_bits = cfg["joystick_bits"].as<int>();
            print_scene_information = cfg["print_scene_information"].as<int>();
            enable_elastic_band = cfg["enable_elastic_band"].as<int>();
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    }
} config;

// 命令行参数解析命名空间
namespace po = boost::program_options;

// 解析命令行参数，可覆盖 YAML 中的 domain_id、interface、scene
// 该函数需在 main 函数起始阶段调用
inline po::variables_map helper(int argc, char** argv)
{
    po::options_description desc("Legbot Mujoco");
    desc.add_options()
        ("help,h", "显示帮助信息")
        ("domain_id,i", po::value<int>(&config.domain_id), "DDS 域 ID；示例：-i 0")
        ("network,n", po::value<std::string>(&config.interface), "DDS 网络接口；示例：-n eth0")
        ("scene,s", po::value<std::filesystem::path>(&config.robot_scene), "机器人场景文件；示例：-s scene_terrain.xml")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        exit(0);
    }

    return vm;
}

}
// 文件用途：策略算法接口与 ONNX Runtime 推理实现。负责加载 .onnx 模型、
// 准备输入张量、执行推理并返回动作向量。
#pragma once

#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <mutex>

namespace isaaclab
{

// 策略算法抽象接口。
class Algorithms
{
public:
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

// ONNX Runtime 推理器：加载策略模型并执行前向推理。
class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path)
    {
        // 初始化 ONNX Runtime 环境与会话选项。
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        // 获取输入名、形状与元素总数。
        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(input_name.release());
        }

        for (const auto& shape : input_shapes) {
            size_t size = 1;
            for (const auto& dim : shape) {
                size *= dim;
            }
            input_sizes.push_back(size);
        }

        // 获取输出名与形状，并按输出维度预分配动作缓冲区。
        Ort::TypeInfo output_type = session->GetOutputTypeInfo(0);
        output_shape = output_type.GetTensorTypeAndShapeInfo().GetShape();
        auto output_name = session->GetOutputNameAllocated(0, allocator);
        output_names.push_back(output_name.release());

        action.resize(output_shape[1]);
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // 校验所有模型输入名都在观察映射中。
        for (const auto& name : input_names) {
            if (obs.find(name) == obs.end()) {
                throw std::runtime_error("Input name " + std::string(name) + " not found in observations.");
            }
        }

        // 创建输入张量（零拷贝，直接引用观察数据）。
        std::vector<Ort::Value> input_tensors;
        for(int i(0); i<input_names.size(); ++i)
        {
            const std::string name_str(input_names[i]);
            auto& input_data = obs.at(name_str);
            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
            input_tensors.push_back(std::move(input_tensor));
        }

        // 执行 ONNX 推理。
        auto output_tensor = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), 1);

        // 将输出数据拷贝到动作缓冲区。
        auto floatarr = output_tensor.front().GetTensorMutableData<float>();
        std::lock_guard<std::mutex> lock(act_mtx_);
        std::memcpy(action.data(), floatarr, output_shape[1] * sizeof(float));
        return action;
    }

private:
    Ort::Env env;                          // ONNX Runtime 环境
    Ort::SessionOptions session_options;   // 会话选项
    std::unique_ptr<Ort::Session> session; // 推理会话
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_names;  // 输入张量名
    std::vector<const char*> output_names; // 输出张量名

    std::vector<std::vector<int64_t>> input_shapes; // 输入形状
    std::vector<int64_t> input_sizes;               // 输入元素数
    std::vector<int64_t> output_shape;              // 输出形状
};
};
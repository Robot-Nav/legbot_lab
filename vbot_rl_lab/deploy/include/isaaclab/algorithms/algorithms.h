// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <mutex>

namespace isaaclab
{

class Algorithms
{
public:
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;
    virtual void reset() {}

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path)
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

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

        for (size_t i = 0; i < session->GetOutputCount(); ++i) {
            Ort::TypeInfo output_type = session->GetOutputTypeInfo(i);
            output_shapes.push_back(output_type.GetTensorTypeAndShapeInfo().GetShape());
            auto output_name = session->GetOutputNameAllocated(i, allocator);
            output_names.push_back(output_name.release());
        }

        // Get first output shape for action
        output_shape = output_shapes[0];
        action.resize(output_shape[1]);

        // Detect MoE+CTS model: has "history" as second input and "new_history" as second output
        has_history_ = false;
        if (input_names.size() >= 2 && std::string(input_names[1]) == "history") {
            has_history_ = true;
            history_shape_ = input_shapes[1];
            size_t history_size = 1;
            for (const auto& dim : history_shape_) {
                history_size *= dim;
            }
            history_buffer_.resize(history_size, 0.0f);
            spdlog::info("OrtRunner: Detected MoE+CTS model with history input, shape=[{},{},{}]",
                         history_shape_[0], history_shape_[1], history_shape_[2]);
        }
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // make sure all input names are in obs
        for (const auto& name : input_names) {
            if (std::string(name) == "history") continue; // history is managed internally
            if (obs.find(name) == obs.end()) {
                throw std::runtime_error("Input name " + std::string(name) + " not found in observations.");
            }
        }

        // Create input tensors
        std::vector<Ort::Value> input_tensors;
        for(int i(0); i<(int)input_names.size(); ++i)
        {
            const std::string name_str(input_names[i]);
            if (name_str == "history" && has_history_) {
                // Use internal history buffer
                auto input_tensor = Ort::Value::CreateTensor<float>(
                    memory_info, history_buffer_.data(), history_buffer_.size(),
                    history_shape_.data(), history_shape_.size());
                input_tensors.push_back(std::move(input_tensor));
            } else {
                auto& input_data = obs.at(name_str);
                auto input_tensor = Ort::Value::CreateTensor<float>(
                    memory_info, input_data.data(), input_sizes[i],
                    input_shapes[i].data(), input_shapes[i].size());
                input_tensors.push_back(std::move(input_tensor));
            }
        }

        // Run the model
        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
            input_tensors.size(), output_names.data(), output_names.size());

        // Copy action output
        auto floatarr = output_tensors.front().GetTensorMutableData<float>();
        std::lock_guard<std::mutex> lock(act_mtx_);
        std::memcpy(action.data(), floatarr, output_shape[1] * sizeof(float));

        // Update history buffer from new_history output (if MoE+CTS model)
        if (has_history_ && output_tensors.size() >= 2) {
            auto new_history_data = output_tensors[1].GetTensorMutableData<float>();
            std::memcpy(history_buffer_.data(), new_history_data, history_buffer_.size() * sizeof(float));
        }

        return action;
    }

    void reset() override
    {
        if (has_history_) {
            std::fill(history_buffer_.begin(), history_buffer_.end(), 0.0f);
        }
    }

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int64_t> input_sizes;
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<int64_t> output_shape;

    // MoE+CTS history state
    bool has_history_;
    std::vector<int64_t> history_shape_;
    std::vector<float> history_buffer_;
};
};
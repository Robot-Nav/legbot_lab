// 文件用途：观察项配置。对每个观察源维护一个带裁剪/缩放的历史缓冲，
// 支持 gym 风格历史展开或直接返回最新值。
#pragma once

#include <deque>
#include <vector>
#include <functional>
#include <numeric>

namespace isaaclab
{

class ManagerBasedRLEnv;

using ObsFunc = std::function<std::vector<float>(ManagerBasedRLEnv*, YAML::Node)>;

struct ObservationTermCfg
{
    YAML::Node params;          // 该观察项的专属参数
    ObsFunc func;               // 观察计算函数
    std::vector<float> clip;    // 裁剪范围
    std::vector<float> scale;   // 缩放系数
    int history_length = 1;     // 历史长度
    bool scale_first = false;   // true 表示先缩放后裁剪

    void reset(std::vector<float> obs)
    {
        for(int i(0); i < history_length; ++i) add(obs);
    }

    void add(std::vector<float> obs)
    {
        for(int j = 0; j < obs.size(); ++j)
        {
            if(scale_first) {
                if(!scale.empty()) obs[j] *= scale[j];
                if (!clip.empty()) {
                    obs[j] = std::clamp(obs[j], clip[0], clip[1]);
                }
            } else {
                if (!clip.empty()) {
                    obs[j] = std::clamp(obs[j], clip[0], clip[1]);
                }
                if(!scale.empty()) obs[j] *= scale[j];
            }
        }
        buff_.push_back(obs);

        if (buff_.size() > history_length) buff_.pop_front();
    }

    const std::vector<float> & get(int n) const { return buff_[n]; }

    const std::vector<float> get() const
    {
        std::vector<float> concatenated;
        for (const auto& entry : buff_) {
            concatenated.insert(concatenated.end(), entry.begin(), entry.end());
        }
        return concatenated;
    }

    const std::size_t size() const { return std::accumulate(buff_.begin(), buff_.end(), 0,
        [](std::size_t sum, const auto& v) { return sum + v.size(); }); }

private:
    // 完整循环缓冲：最新条目在尾部，最旧条目在头部。
    std::deque<std::vector<float>> buff_;
};

};
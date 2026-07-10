// 文件用途：分段线性插值器。FixStand 状态按时间序列 ts 与关节角度序列 qs 平滑过渡，
// 避免机器人从当前姿态直接跳变到目标姿态造成冲击。
#pragma once

#include <vector>
#include <cassert>

// 一维时间 t 对多维关节角度进行分段线性插值。
// ts 必须单调递增，ys 与 ts 长度相同，返回向量维度与 ys[0] 一致。
inline std::vector<float> linear_interpolate(float t, const std::vector<float>& ts, const std::vector<std::vector<float>>& ys)
{
    assert(ts.size() == ys.size() && !ys.empty() && ts.size() > 1 && ys[0].size() > 0);

    // 边界外推：小于首时刻用首姿态，大于末时刻用末姿态，保证插值稳定。
    if (t <= ts[0]) return ys[0];
    if (t >= ts[ts.size() - 1]) return ys[ts.size() - 1];

    for (int i = 0; i < ts.size() - 1; ++i)
    {
        if (t >= ts[i] && t <= ts[i + 1])
        {
            float alpha = (t - ts[i]) / (ts[i + 1] - ts[i]);
            std::vector<float> result(ys[i].size());
            for (int j = 0; j < ys[i].size(); ++j)
            {
                result[j] = ys[i][j] * (1 - alpha) + ys[i + 1][j] * alpha;
            }
            return result;
        }
    }

    return std::vector<float>(ys[0].size(), 0.0f); // 理论上不会执行到此处
}

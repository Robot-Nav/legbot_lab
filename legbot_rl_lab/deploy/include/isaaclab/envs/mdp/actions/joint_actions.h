// 文件用途：关节动作项。将策略网络输出按配置进行缩放、偏移与裁剪，得到目标关节位置/速度。
#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/manager/action_manager.h"

namespace isaaclab
{

// 基础关节动作：对原始动作依次进行 scale、offset、clip。
class JointAction : public ActionTerm
{
public:
    JointAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :ActionTerm(cfg, env)
    {
        if(cfg["joint_ids"].IsNull()) {
            _action_dim = env->robot->data.joint_ids_map.size();
        } else {
            _joint_ids = cfg["joint_ids"].as<std::vector<int>>();
            _action_dim = _joint_ids.size();
        }
        _raw_actions.resize(_action_dim, 0.0f);
        _processed_actions.resize(_action_dim, 0.0f);
        if(!cfg["scale"].IsNull()) {
            _scale = cfg["scale"].as<std::vector<float>>();
        }
        if(!cfg["offset"].IsNull()) {
            _offset = cfg["offset"].as<std::vector<float>>();
        }
        if(!cfg["clip"].IsNull()) {
            _clip = cfg["clip"].as<std::vector<std::vector<float> >>();
        }
    }

    virtual void process_actions(std::vector<float> actions)
    {
        _raw_actions = actions;
        for(int i(0); i<_action_dim; ++i)
        {
            if(!_scale.empty()) {
                _processed_actions[i] = _raw_actions[i] * _scale[i];
            } else {
                _processed_actions[i] = _raw_actions[i];
            }
            if(!_offset.empty()) {
                _processed_actions[i] += _offset[i];
            }
        }
        if(!_clip.empty())
        {
            for(int i(0); i<_action_dim; ++i) {
                _processed_actions[i] = std::clamp(_processed_actions[i], _clip[i][0], _clip[i][1]);
            }
        }
    }


    int action_dim() 
    {
        return _action_dim;
    }

    std::vector<float> raw_actions() 
    {
        return _raw_actions;
    }
    
    std::vector<float> processed_actions() 
    {
        return _processed_actions;
    }

    void reset()
    {
        _raw_actions.assign(_action_dim, 0.0f);
    }

protected:
    int _action_dim;                    // 动作维度
    std::vector<int> _joint_ids;        // 受控关节索引（当前未使用）

    std::vector<float> _raw_actions;       // 网络原始输出
    std::vector<float> _processed_actions; // 缩放/偏移/裁剪后的目标量

    std::vector<float> _scale;          // 缩放系数
    std::vector<float> _offset;         // 偏移量
    std::vector<std::vector<float> > _clip; // 每维上下限
};


// 关节位置动作：输出直接作为目标关节角度。
class JointPositionAction : public JointAction
{
public:
    JointPositionAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :JointAction(cfg, env)
    {
    }
};

// 关节速度动作：输出作为目标关节速度（当前实现与位置动作相同，由外层解释）。
class JointVelocityAction : public JointAction
{
public:
    JointVelocityAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :JointAction(cfg, env)
    {
    }
};

REGISTER_ACTION(JointPositionAction);
REGISTER_ACTION(JointVelocityAction);

};
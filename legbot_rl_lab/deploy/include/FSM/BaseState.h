// 文件用途：FSM 基类与状态工厂注册。所有具体状态继承 BaseState，
// 并通过 REGISTER_FSM 宏在启动时自动注册到全局工厂表中。
#pragma once

#include <boost/bimap.hpp>
#include <string>
#include <any>
#include <utility>

// 状态 ID 与状态名字符串之间的双向映射，用于配置解析与日志输出。
inline boost::bimap<int, std::string> FSMStringMap;

// FSM 状态基类。每个状态拥有 enter / pre_run / run / post_run / exit 生命周期，
// 并维护一组切换条件 registered_checks。
class BaseState
{
public:
    BaseState(int state, std::string state_string) : state_(state) 
    {
        FSMStringMap.insert({state, state_string});
    }

    virtual void enter() {}

    virtual void pre_run() {}
    virtual void run() {}
    virtual void post_run() {}

    virtual void exit() {}

    std::string getStateString() { return FSMStringMap.left.at(state_); }
    int getState() {return state_; }
    bool isState(int state) { return state_ == state; }
    std::vector<std::pair<std::function<bool()>, int>> registered_checks;
private:
    int state_;
};

using FsmFactory = std::function<std::shared_ptr<BaseState>(int, std::string)>;
using FsmMap     = std::unordered_map<std::string, FsmFactory>;

inline FsmMap& getFsmMap() {
    static FsmMap fsmMap;
    return fsmMap;
}

// 自动注册状态到全局工厂表，使 CtrlFSM 可通过状态类型名动态创建实例。
#define REGISTER_FSM(Derived) \
    inline std::shared_ptr<BaseState> __factory_##Derived(int s, std::string ss) {      \
        return std::make_shared<Derived>(s, ss);                                        \
    }                                                                                   \
    inline struct __registrar_##Derived {                                               \
        __registrar_##Derived() {                                                       \
            getFsmMap()[#Derived] = __factory_##Derived;                                \
        }                                                                               \
    } __registrar_instance_##Derived;                                                   \
    

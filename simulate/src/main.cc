// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// MuJoCo 仿真主程序
// 加载机器人模型、启动物理步进线程、渲染 UI，并通过 DDS 桥接线程与宇树 Go2 控制栈通信。

// 临时 trick：让 glfw_adapter 的私有成员 window_ 可被外部访问，用于设置键盘回调
#define private public
#include "glfw_adapter.h"
#undef private

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>

#include <mujoco/mujoco.h>
#include "simulate.h"
#include "array_safety.h"
#include "legbot_bridge.h"
#include "param.h"

// MuJoCo 渲染窗口全局句柄，桥接线程与键盘回调均依赖它
GLFWwindow* g_sim_window = nullptr;

// MuJoCo 插件库所在目录，用于自动加载自定义插件
#define MUJOCO_PLUGIN_DIR "mujoco_plugin"

extern "C"
{
#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <sys/errno.h>
#include <unistd.h>
#endif
}

// 弹性绳约束：在 base 与固定点之间施加弹簧阻尼力，用于防止仿真机器人倾倒或快速漂移
class ElasticBand
{
public:
  ElasticBand(){};
  void Advance(std::vector<double> x, std::vector<double> dx)
  {
    std::vector<double> delta_x = {0.0, 0.0, 0.0};
    delta_x[0] = point_[0] - x[0];
    delta_x[1] = point_[1] - x[1];
    delta_x[2] = point_[2] - x[2];
    double distance = sqrt(delta_x[0] * delta_x[0] + delta_x[1] * delta_x[1] + delta_x[2] * delta_x[2]);

    std::vector<double> direction = {0.0, 0.0, 0.0};
    direction[0] = delta_x[0] / distance;
    direction[1] = delta_x[1] / distance;
    direction[2] = delta_x[2] / distance;

    // 计算 base 沿绳方向的速度投影
    double v = dx[0] * direction[0] + dx[1] * direction[1] + dx[2] * direction[2];

    // 弹簧阻尼力 = 刚度 * (距离 - 绳长) - 阻尼 * 速度，再投影到方向
    f_[0] = (stiffness_ * (distance - length_) - damping_ * v) * direction[0];
    f_[1] = (stiffness_ * (distance - length_) - damping_ * v) * direction[1];
    f_[2] = (stiffness_ * (distance - length_) - damping_ * v) * direction[2];
  }


  double stiffness_ = 200;                 // 弹性绳刚度
  double damping_ = 100;                   // 弹性绳阻尼
  std::vector<double> point_ = {0, 0, 3};  // 固定点位置
  double length_ = 0.0;                    // 绳自然长度，可通过键盘动态调整
  bool enable_ = true;                     // 是否启用
  std::vector<double> f_ = {0, 0, 0};      // 当前作用在 base 上的力
};
inline ElasticBand elastic_band;


namespace
{
  namespace mj = ::mujoco;
  namespace mju = ::mujoco::sample_util;

  // 物理与渲染同步参数
  const double syncMisalign = 0.1;       // CPU 与仿真时间偏差超过此值时重新同步（单位：仿真秒）
  const double simRefreshFraction = 0.7; // 每帧刷新时间内可用于物理步进的比例
  const int kErrorLength = 1024;         // 模型加载错误信息缓冲区长度

  // MuJoCo 模型与数据指针
  mjModel *m = nullptr;
  mjData *d = nullptr;

  // 控制量噪声缓存，用于 Ornstein-Uhlenbeck 过程
  mjtNum *ctrlnoise = nullptr;

  using Seconds = std::chrono::duration<double>;

  //---------------------------------------- 插件处理 -----------------------------------------

  // 获取当前可执行文件所在目录，用于定位 config.yaml 和 mujoco_plugin 目录
  std::string getExecutableDir()
  {
#if defined(_WIN32) || defined(__CYGWIN__)
    constexpr char kPathSep = '\\';
    std::string realpath = [&]() -> std::string
    {
      std::unique_ptr<char[]> realpath(nullptr);
      DWORD buf_size = 128;
      bool success = false;
      while (!success)
      {
        realpath.reset(new (std::nothrow) char[buf_size]);
        if (!realpath)
        {
          std::cerr << "cannot allocate memory to store executable path\n";
          return "";
        }

        DWORD written = GetModuleFileNameA(nullptr, realpath.get(), buf_size);
        if (written < buf_size)
        {
          success = true;
        }
        else if (written == buf_size)
        {
          // 缓冲区不足，扩容后重试
          buf_size *= 2;
        }
        else
        {
          std::cerr << "failed to retrieve executable path: " << GetLastError() << "\n";
          return "";
        }
      }
      return realpath.get();
    }();
#else
    constexpr char kPathSep = '/';
#if defined(__APPLE__)
    std::unique_ptr<char[]> buf(nullptr);
    {
      std::uint32_t buf_size = 0;
      _NSGetExecutablePath(nullptr, &buf_size);
      buf.reset(new char[buf_size]);
      if (!buf)
      {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }
      if (_NSGetExecutablePath(buf.get(), &buf_size))
      {
        std::cerr << "unexpected error from _NSGetExecutablePath\n";
      }
    }
    const char *path = buf.get();
#else
    const char *path = "/proc/self/exe";
#endif
    std::string realpath = [&]() -> std::string
    {
      std::unique_ptr<char[]> realpath(nullptr);
      std::uint32_t buf_size = 128;
      bool success = false;
      while (!success)
      {
        realpath.reset(new (std::nothrow) char[buf_size]);
        if (!realpath)
        {
          std::cerr << "cannot allocate memory to store executable path\n";
          return "";
        }

        std::size_t written = readlink(path, realpath.get(), buf_size);
        if (written < buf_size)
        {
          realpath.get()[written] = '\0';
          success = true;
        }
        else if (written == -1)
        {
          if (errno == EINVAL)
          {
            // 路径本身不是符号链接，直接使用
            return path;
          }

          std::cerr << "error while resolving executable path: " << strerror(errno) << '\n';
          return "";
        }
        else
        {
          // 缓冲区不足，扩容后重试
          buf_size *= 2;
        }
      }
      return realpath.get();
    }();
#endif

    if (realpath.empty())
    {
      return "";
    }

    for (std::size_t i = realpath.size() - 1; i > 0; --i)
    {
      if (realpath.c_str()[i] == kPathSep)
      {
        return realpath.substr(0, i);
      }
    }

    // 未找到路径分隔符，避免向上遍历根目录
    return "";
  }

  // 扫描并加载 mujoco_plugin 目录下的动态库插件
  void scanPluginLibraries()
  {
    // 打印直接链接到可执行文件中的内置插件
    int nplugin = mjp_pluginCount();
    if (nplugin)
    {
      std::printf("Built-in plugins:\n");
      for (int i = 0; i < nplugin; ++i)
      {
        std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
      }
    }

    // 根据平台选择路径分隔符
#if defined(_WIN32) || defined(__CYGWIN__)
    const std::string sep = "\\";
#else
    const std::string sep = "/";
#endif

    // 打开可执行文件所在目录下的 mujoco_plugin 目录
    const std::string executable_dir = getExecutableDir();
    if (executable_dir.empty())
    {
      return;
    }

    const std::string plugin_dir = getExecutableDir() + sep + MUJOCO_PLUGIN_DIR;
    mj_loadAllPluginLibraries(
        plugin_dir.c_str(), +[](const char *filename, int first, int count)
                            {
        std::printf("Plugins registered by library '%s':\n", filename);
        for (int i = first; i < first + count; ++i) {
          std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
        } });
  }

  //------------------------------------------- 仿真步进 -------------------------------------------

  // 加载并编译 MuJoCo 模型，支持 .mjb 二进制或 .xml 文件
  mjModel *LoadModel(const char *file, mj::Simulate &sim)
  {
    // 拷贝到定长数组以满足 mju::strlen_arr 的编译要求
    char filename[mj::Simulate::kMaxFilenameLength];
    mju::strcpy_arr(filename, file);

    if (!filename[0])
    {
      return nullptr;
    }

    // 根据后缀选择二进制或 XML 加载方式
    char loadError[kErrorLength] = "";
    mjModel *mnew = 0;
    if (mju::strlen_arr(filename) > 4 &&
        !std::strncmp(filename + mju::strlen_arr(filename) - 4, ".mjb",
                      mju::sizeof_arr(filename) - mju::strlen_arr(filename) + 4))
    {
      mnew = mj_loadModel(filename, nullptr);
      if (!mnew)
      {
        mju::strcpy_arr(loadError, "could not load binary model");
      }
    }
    else
    {
      mnew = mj_loadXML(filename, nullptr, loadError, kErrorLength);
      // 去除错误信息末尾的换行符，便于 UI 显示
      if (loadError[0])
      {
        int error_length = mju::strlen_arr(loadError);
        if (loadError[error_length - 1] == '\n')
        {
          loadError[error_length - 1] = '\0';
        }
      }
    }

    mju::strcpy_arr(sim.load_error, loadError);

    if (!mnew)
    {
      std::printf("%s\n", loadError);
      return nullptr;
    }

    // 编译警告时暂停仿真，待用户确认
    if (loadError[0])
    {
      // 后续 mj_forward 会再次打印该警告
      std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
      sim.run = 0;
    }

    return mnew;
  }

  // 物理步进后台线程：与主线程渲染并行，维持软实时仿真
  void PhysicsLoop(mj::Simulate &sim)
  {
    // CPU 时间与仿真时间的同步锚点
    std::chrono::time_point<mj::Simulate::Clock> syncCPU;
    mjtNum syncSim = 0;

    // 循环直到 UI 请求退出
    while (!sim.exitrequest.load())
    {
      // 处理拖放文件加载请求
      if (sim.droploadrequest.load())
      {
        sim.LoadMessage(sim.dropfilename);
        mjModel *mnew = LoadModel(sim.dropfilename, sim);
        sim.droploadrequest.store(false);

        mjData *dnew = nullptr;
        if (mnew)
          dnew = mj_makeData(mnew);
        if (dnew)
        {
          sim.Load(mnew, dnew, sim.dropfilename);

          mj_deleteData(d);
          mj_deleteModel(m);

          m = mnew;
          d = dnew;
          mj_forward(m, d);

          // 按新模型执行器数量重新分配控制噪声缓存
          free(ctrlnoise);
          ctrlnoise = (mjtNum *)malloc(sizeof(mjtNum) * m->nu);
          mju_zero(ctrlnoise, m->nu);
        }
        else
        {
          sim.LoadMessageClear();
        }
      }

      // 处理 UI 文件加载请求
      if (sim.uiloadrequest.load())
      {
        sim.uiloadrequest.fetch_sub(1);
        sim.LoadMessage(sim.filename);
        mjModel *mnew = LoadModel(sim.filename, sim);
        mjData *dnew = nullptr;
        if (mnew)
          dnew = mj_makeData(mnew);
        if (dnew)
        {
          sim.Load(mnew, dnew, sim.filename);

          mj_deleteData(d);
          mj_deleteModel(m);

          m = mnew;
          d = dnew;
          mj_forward(m, d);

          // 按新模型执行器数量重新分配控制噪声缓存
          free(ctrlnoise);
          ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
          mju_zero(ctrlnoise, m->nu);
        }
        else
        {
          sim.LoadMessageClear();
        }
      }

      // 让出时间片给主线程；busywait 模式时序更准但功耗更高
      if (sim.run && sim.busywait)
      {
        std::this_thread::yield();
      }
      else
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      {
        // 加锁保护模型数据，避免与渲染线程并发访问
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        if (m)
        {
          // 运行态：按软实时策略推进物理仿真
          if (sim.run)
          {
            bool stepped = false;

            // 记录本次循环开始时的 CPU 时间
            const auto startCPU = mj::Simulate::Clock::now();

            // 计算自上次同步以来经过的 CPU 时间和仿真时间
            const auto elapsedCPU = startCPU - syncCPU;
            double elapsedSim = d->time - syncSim;

            // 注入控制噪声，模拟真实执行器扰动
            if (sim.ctrl_noise_std)
            {
              // Ornstein-Uhlenbeck 离散化：rate 为衰减系数，scale 为噪声幅值
              mjtNum rate = mju_exp(-m->opt.timestep / mju_max(sim.ctrl_noise_rate, mjMINVAL));
              mjtNum scale = sim.ctrl_noise_std * mju_sqrt(1 - rate * rate);

              for (int i = 0; i < m->nu; i++)
              {
                ctrlnoise[i] = rate * ctrlnoise[i] + scale * mju_standardNormal(nullptr);
                d->ctrl[i] = ctrlnoise[i];
              }
            }

            // 用户选择的慢放倍数
            double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

            // 判断 CPU 与仿真时间是否显著偏离
            bool misaligned =
                mju_abs(Seconds(elapsedCPU).count() / slowdown - elapsedSim) > syncMisalign;

            // 失步或首次运行时重置同步锚点并单步推进
            if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 ||
                misaligned || sim.speed_changed)
            {
              syncCPU = startCPU;
              syncSim = d->time;
              sim.speed_changed = false;

              // 先走一步，后续循环再处理精确时序
              mj_step(m, d);
              stepped = true;
            }

            // 同步时：在刷新周期内多步推进，直到仿真时间追上 CPU 时间
            else
            {
              bool measured = false;
              mjtNum prevSim = d->time;

              double refreshTime = simRefreshFraction / sim.refresh_rate;

              while (Seconds((d->time - syncSim) * slowdown) < mj::Simulate::Clock::now() - syncCPU &&
                     mj::Simulate::Clock::now() - startCPU < Seconds(refreshTime))
              {
                // 在第一次有效步进前测量实际慢放倍数
                if (!measured && elapsedSim)
                {
                  sim.measured_slowdown =
                      std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
                  measured = true;
                }

                // 在 base 上施加弹性绳外力
                if (param::config.enable_elastic_band == 1)
                {
                  if (elastic_band.enable_)
                  {
                    std::vector<double> x = {d->qpos[0], d->qpos[1], d->qpos[2]};
                    std::vector<double> dx = {d->qvel[0], d->qvel[1], d->qvel[2]};

                    elastic_band.Advance(x, dx);

                    d->xfrc_applied[param::config.band_attached_link] = elastic_band.f_[0];
                    d->xfrc_applied[param::config.band_attached_link + 1] = elastic_band.f_[1];
                    d->xfrc_applied[param::config.band_attached_link + 2] = elastic_band.f_[2];
                  }
                }

                mj_step(m, d);
                stepped = true;

                // 仿真时间回退说明发生重置，退出本帧步进
                if (d->time < prevSim)
                {
                  break;
                }
              }
            }

            // 将最新状态加入历史缓存，支持 UI 回滚与慢放
            if (stepped)
            {
              sim.AddToHistory();
            }
          }

          // 暂停态：仅做正向运动学更新，保证渲染和滑条同步
          else
          {
            mj_forward(m, d);
            sim.speed_changed = true;
          }
        }
      } // 释放锁
    }
  }
} // namespace

//-------------------------------------- 物理线程 --------------------------------------------

void PhysicsThread(mj::Simulate *sim, const char *filename)
{
  // 若命令行/配置文件指定了模型文件则直接加载，否则等待拖放
  if (filename != nullptr)
  {
    sim->LoadMessage(filename);
    m = LoadModel(filename, *sim);
    if (m)
      d = mj_makeData(m);
    if (d)
    {
      sim->Load(m, d, filename);
      mj_forward(m, d);

      // 分配控制噪声缓存
      free(ctrlnoise);
      ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
      mju_zero(ctrlnoise, m->nu);
    }
    else
    {
      sim->LoadMessageClear();
    }
  }

  PhysicsLoop(*sim);

  // 清理分配的资源
  free(ctrlnoise);
  mj_deleteData(d);
  mj_deleteModel(m);

  exit(0);
}

// DDS 桥接线程：等待 MuJoCo 数据就绪后初始化 DDS 并启动 VBotBridge
void *UnitreeSdk2BridgeThread(void *arg)
{
  // 等待物理线程创建好 mjData
  while (true)
  {
    if (d)
    {
      std::cout << "Mujoco data is prepared" << std::endl;
      break;
    }
    usleep(500000);
  }

  // 使用 YAML/命令行指定的域 ID 和网络接口初始化 DDS
  unitree::robot::ChannelFactory::Instance()->Init(param::config.domain_id, param::config.interface);


  // 查找 base 或 base_link 刚体，计算弹性绳作用索引
  // xfrc_applied 每个刚体占 6 个 double（力 + 力矩），故起始索引为 6 * body_id
  int body_id = mj_name2id(m, mjOBJ_BODY, "base");
  if (body_id < 0) {
    body_id = mj_name2id(m, mjOBJ_BODY, "base_link");
  }
  param::config.band_attached_link = 6 * body_id;

  auto interface = std::make_unique<VBotBridge>(m, d);
  interface->start();

  while (true)
  {
    sleep(1);
  }
}
//------------------------------------------ 主函数 --------------------------------------------------

// macOS Rosetta 2 不支持时的弹窗提示机制
#if defined(__APPLE__) && defined(__AVX__)
extern void DisplayErrorDialogBox(const char *title, const char *msg);
static const char *rosetta_error_msg = nullptr;
__attribute__((used, visibility("default"))) extern "C" void _mj_rosettaError(const char *msg)
{
  rosetta_error_msg = msg;
}
#endif

// 用户键盘回调：在默认回调基础上增加弹性绳和重置快捷键
static GLFWkeyfun s_mujoco_key_callback = nullptr;

void user_key_cb(GLFWwindow* window, int key, int scancode, int act, int mods) {
  if (s_mujoco_key_callback) {
    s_mujoco_key_callback(window, key, scancode, act, mods);
  }

  if (act==GLFW_PRESS)
  {
    if(param::config.enable_elastic_band == 1) {
      if (key==GLFW_KEY_9) {
        elastic_band.enable_ = !elastic_band.enable_;
      } else if (key==GLFW_KEY_7 || key==GLFW_KEY_UP) {
        elastic_band.length_ -= 0.1;
      } else if (key==GLFW_KEY_8 || key==GLFW_KEY_DOWN) {
        elastic_band.length_ += 0.1;
      }
    }
    if(key==GLFW_KEY_BACKSPACE) {
      mj_resetData(m, d);
      mj_forward(m, d);
    }
  }
}

int main(int argc, char **argv)
{

  // macOS Rosetta 2 不支持运行，直接弹窗报错
#if defined(__APPLE__) && defined(__AVX__)
  if (rosetta_error_msg)
  {
    DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
    std::exit(1);
  }
#endif

  // 打印 MuJoCo 版本并校验头文件与库版本一致
  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER != mj_version())
  {
    mju_error("Headers and library have different versions");
  }

  // 加载 mujoco_plugin 目录下的插件
  scanPluginLibraries();

  // 初始化可视化相关结构体
  mjvCamera cam;
  mjv_defaultCamera(&cam);

  mjvOption opt;
  mjv_defaultOption(&opt);

  mjvPerturb pert;
  mjv_defaultPerturb(&pert);

  // 加载仿真配置：先读 YAML，再用命令行参数覆盖
  std::filesystem::path proj_dir = std::filesystem::path(getExecutableDir()).parent_path();
  param::config.load_from_yaml(proj_dir / "config.yaml");
  param::helper(argc, argv);
  if(param::config.robot_scene.is_relative()) {
    param::config.robot_scene = proj_dir.parent_path() / "legbot" / "xmls" / param::config.robot_scene;
  }

  // 创建仿真 UI 对象，负责渲染与交互
  auto sim = std::make_unique<mj::Simulate>(
    std::make_unique<mj::GlfwAdapter>(),
    &cam, &opt, &pert, /* is_passive = */ false);

  g_sim_window = static_cast<mj::GlfwAdapter*>(sim->platform_ui.get())->window_;

  // 启动 DDS 桥接线程，必须在物理线程前创建以便及时接管数据
  std::thread unitree_thread(UnitreeSdk2BridgeThread, nullptr);

  // 启动物理仿真线程
  std::thread physicsthreadhandle(&PhysicsThread, sim.get(), param::config.robot_scene.c_str());
  // 设置键盘回调并进入阻塞式渲染主循环
  s_mujoco_key_callback = glfwSetKeyCallback(g_sim_window, user_key_cb);
  sim->RenderLoop();
  physicsthreadhandle.join();

  pthread_exit(NULL);
  return 0;
}

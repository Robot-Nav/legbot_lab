# 环境配置解析工具：根据任务名从注册表中加载并覆盖默认配置。
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def parse_env_cfg(
    task_name: str,
    device: str = "cuda:0",
    num_envs: int | None = None,
    use_fabric: bool | None = None,
    entry_point_key: str = "env_cfg_entry_point",
) -> ManagerBasedRLEnvCfg | DirectRLEnvCfg:
    """解析并覆盖指定任务的配置。

    参数：
        task_name: 任务/环境名称。
        device: 运行仿真的设备，默认为 "cuda:0"。
        num_envs: 环境数量。为 None 时保持默认值不变。
        use_fabric: 是否启用 Fabric 接口。为 False 时通过 USD 读写，便于可视化但速度较慢。
                    为 None 时保持默认值不变。
        entry_point_key: 注册表中配置入口的键名。

    返回：
        解析后的环境配置对象。

    异常：
        RuntimeError: 若配置不是类，则抛出异常（本函数要求使用类作为配置）。
    """
    # 加载默认配置
    cfg = load_cfg_from_registry(task_name, entry_point_key)

    # 要求配置必须是类而非字典
    if isinstance(cfg, dict):
        raise RuntimeError(f"任务 '{task_name}' 的配置不是类，请提供类配置。")

    # 设置仿真设备
    cfg.sim.device = device
    # 是否启用 Fabric
    if use_fabric is not None:
        cfg.sim.use_fabric = use_fabric
    # 覆盖环境数量
    if num_envs is not None:
        cfg.scene.num_envs = num_envs

    return cfg

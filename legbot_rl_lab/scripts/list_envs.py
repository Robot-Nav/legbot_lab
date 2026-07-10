"""
列出 Isaac Lab 中所有可用环境的脚本。

遍历已注册环境并以表格形式输出环境名称、入口点与配置文件路径。
所有环境注册在 `unitree_rl_lab` 扩展中，名称以 `Unitree` 开头。
"""

"""首先启动 Isaac Sim 仿真器。"""


import importlib
import pathlib
import pkgutil
import sys


def _walk_packages(
    path: str | None = None,
    prefix: str = "",
    onerror=None,
):
    """递归遍历指定路径下的所有模块，生成 ModuleInfo。

    说明：
        本函数基于 ``pkgutil.walk_packages`` 修改而来，用于导入项目任务包。
    """

    def seen(p, m={}):
        if p in m:
            return True
        m[p] = True  # noqa: R503

    for info in pkgutil.iter_modules(path, prefix):

        # 返回当前模块信息
        yield info

        if info.ispkg:
            try:
                __import__(info.name)
            except Exception:
                if onerror is not None:
                    onerror(info.name)
                else:
                    raise
            else:
                path = getattr(sys.modules[info.name], "__path__", None) or []

                # 跳过已遍历的路径，避免重复
                path = [p for p in path if not seen(p)]

                yield from _walk_packages(path, info.name + ".", onerror)


def import_packages():
    """导入 unitree_rl_lab 任务包，完成环境注册。"""
    sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}/source/unitree_rl_lab/unitree_rl_lab/tasks/")
    for package in ["locomotion.robots"]:
        package = importlib.import_module(package)
        for _ in _walk_packages(package.__path__, package.__name__ + "."):
            pass
    sys.path.pop(0)


import_packages()

"""后续逻辑。"""

import gymnasium as gym
from prettytable import PrettyTable


def main():
    """以表格形式打印 unitree_rl_lab 扩展中注册的所有环境。"""
    # 创建输出表格
    table = PrettyTable(["序号", "任务名称", "入口点", "配置文件"])
    table.title = "Unitree RL Lab 可用环境"
    # 设置列对齐方式
    table.align["任务名称"] = "l"
    table.align["入口点"] = "l"
    table.align["配置文件"] = "l"

    # 环境序号
    index = 0
    # 遍历 gym 注册表，筛选 Unitree 相关环境
    for task_spec in gym.registry.values():
        if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
            # 添加环境详情到表格
            table.add_row([index + 1, task_spec.id, task_spec.entry_point, task_spec.kwargs["env_cfg_entry_point"]])
            # 序号递增
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        # 执行主函数
        main()
    except Exception as e:
        raise e

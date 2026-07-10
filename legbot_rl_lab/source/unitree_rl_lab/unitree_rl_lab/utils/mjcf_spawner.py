# MJCF 资源加载工具：提供从 MJCF 文件生成 USD Prim 的 spawn 函数与配置类。
from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.sim import converters
from isaaclab.sim.spawners.from_files.from_files_cfg import FileCfg
from isaaclab.utils import configclass

import isaacsim.core.utils.prims as prim_utils
from pxr import Usd

from isaaclab.sim.utils import clone
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file


@clone
def spawn_from_mjcf(
    prim_path: str,
    cfg: MjcfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """将 MJCF 文件转换为 USD 并 spawn 到指定 prim 路径。

    参数：
        prim_path: 目标 prim 路径。
        cfg: MJCF 文件配置。
        translation: 平移偏移，可选。
        orientation: 四元数旋转偏移，可选。
        **kwargs: 额外参数，透传给内部 spawn 函数。

    返回：
        生成的 USD Prim。
    """
    mjcf_loader = converters.MjcfConverter(cfg)
    return _spawn_from_usd_file(prim_path, mjcf_loader.usd_path, cfg, translation, orientation)


@configclass
class MjcfFileCfg(FileCfg, converters.MjcfConverterCfg):
    """MJCF 文件 spawn 配置：指定使用 spawn_from_mjcf 函数。"""

    func: Callable = spawn_from_mjcf

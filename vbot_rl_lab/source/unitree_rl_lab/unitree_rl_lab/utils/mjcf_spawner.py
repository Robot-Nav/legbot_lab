from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.sim import converters
from isaaclab.sim.spawners.from_files.from_files_cfg import FileCfg
from isaaclab.utils import configclass

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
    mjcf_loader = converters.MjcfConverter(cfg)
    return _spawn_from_usd_file(prim_path, mjcf_loader.usd_path, cfg, translation, orientation)


@configclass
class MjcfFileCfg(FileCfg, converters.MjcfConverterCfg):
    func: Callable = spawn_from_mjcf

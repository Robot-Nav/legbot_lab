# Height-field formulas adapted from Isaac Lab, Copyright (c) 2022-2025,
# The Isaac Lab Project Developers. All rights reserved. BSD-3-Clause.
# See ISAACLAB_LICENSE.txt for the full license and disclaimer.

"""Standalone MuJoCo equivalents of Legbot's eight sampled training terrains.

Parameters mirror ``tasks/go2/mdp/terrains.py:TERRAIN_CFG``. No Isaac Sim
installation is needed. Smooth terrain uses MuJoCo height fields; stairs and
discrete obstacles use merged box columns to preserve vertical collision faces.
The latter approximate Isaac Lab's slope-corrected triangle meshes on a 0.1 m
grid, rather than using MuJoCo's convex hull of a concave terrain mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


TRAINING_TERRAINS = (
    "flat", "wave", "slope_up", "slope_down", "rough_slope",
    "stairs_up", "stairs_down", "obstacles",
)
PATCH_SIZE = 9.0
HORIZONTAL_SCALE = 0.1
VERTICAL_SCALE = 0.005
BORDER_WIDTH = 0.5


@dataclass
class Terrain:
    name: str
    heights: np.ndarray  # x, y axes, in metres, including the zero-height border
    origin_z: float
    difficulty: float
    seed: int


def generate_terrain(name: str, difficulty: float = 0.5, seed: int = 0,
                     *, stair_height: float | None = None) -> Terrain:
    """Use the training interpolation, quantization, border and reset origin.

    These formulas follow Isaac Lab's height-field generators (BSD-3-Clause;
    see ISAACLAB_LICENSE.txt). A local RandomState keeps generation reproducible
    without changing NumPy's global random state. An explicit stair_height fixes
    the stair risers in metres, quantized to VERTICAL_SCALE, independently of
    difficulty; the other terrains continue to use their training parameters.
    """
    if name not in TRAINING_TERRAINS:
        raise ValueError(f"Unknown training terrain: {name}")
    if not np.isfinite(difficulty) or not 0 <= difficulty <= 1:
        raise ValueError("Terrain difficulty must be between 0 and 1.")
    if stair_height is not None and (not np.isfinite(stair_height) or stair_height < VERTICAL_SCALE):
        raise ValueError(f"Stair height must be finite and at least {VERTICAL_SCALE} m.")
    rng = np.random.RandomState(seed)
    count = round(PATCH_SIZE / HORIZONTAL_SCALE) + 1
    border = int(BORDER_WIDTH / HORIZONTAL_SCALE) + 1
    n = count - 2 * border
    raw = np.zeros((n, n), dtype=np.int16)

    if name in ("slope_up", "slope_down", "rough_slope"):
        slope = 0.1 + difficulty * (0.568 - 0.1)
        if name == "slope_up":
            slope *= -1
        maximum = int(slope * (n * HORIZONTAL_SCALE) / 2 / VERTICAL_SCALE)
        center = n // 2
        profile = (center - np.abs(center - np.arange(n))) / center
        surface = maximum * profile[:, None] * profile[None, :]
        platform_corner = center - int(3.0 / HORIZONTAL_SCALE / 2)
        platform_z = surface[platform_corner, platform_corner]
        raw = np.rint(np.clip(surface, min(0, platform_z), max(0, platform_z))).astype(np.int16)

    elif name in ("stairs_up", "stairs_down"):
        step_height = 0.05 + difficulty * (0.257 - 0.05) if stair_height is None else stair_height
        if name == "stairs_up":
            step_height *= -1
        step_pixels = (int(step_height / VERTICAL_SCALE) if stair_height is None
                       else round(step_height / VERTICAL_SCALE))
        step_width = int(0.31 / HORIZONTAL_SCALE)
        start, end, height = 0, n, 0
        while end - start > int(3.0 / HORIZONTAL_SCALE):
            start += step_width
            end -= step_width
            height += step_pixels
            raw[start:end, start:end] = height

    elif name == "obstacles":
        height = int((0.05 + difficulty * (0.275 - 0.05)) / VERTICAL_SCALE)
        widths = np.arange(int(1.0 / HORIZONTAL_SCALE), int(2.0 / HORIZONTAL_SCALE), 4)
        positions = np.arange(0, n, 4)
        for _ in range(20):
            top = rng.choice([-height, -height // 2, height // 2, height])
            width, length = int(rng.choice(widths)), int(rng.choice(widths))
            x = min(int(rng.choice(positions)), n - width)
            y = min(int(rng.choice(positions)), n - length)
            raw[x:x + width, y:y + length] = top
        platform = int(3.0 / HORIZONTAL_SCALE)
        lo, hi = (n - platform) // 2, (n + platform) // 2
        raw[lo:hi, lo:hi] = 0

    elif name == "wave":
        amplitude = int(0.5 * (0.1 + difficulty * (0.28 - 0.1)) / VERTICAL_SCALE)
        phase = np.arange(n) * (2 * np.pi / (n / 5))
        raw = np.rint(amplitude * (np.sin(phase[:, None]) + np.cos(phase[None, :]))).astype(np.int16)

    if name in ("wave", "rough_slope"):
        # Same coarse noise sampling and cubic interpolation as training.
        from scipy.interpolate import RectBivariateSpline

        coarse_count = int((n * HORIZONTAL_SCALE) / 0.2)
        noise = rng.choice(np.arange(-10, 11), size=(coarse_count, coarse_count))
        coarse = np.linspace(0, n * HORIZONTAL_SCALE**2, coarse_count)
        fine = np.linspace(0, n * HORIZONTAL_SCALE**2, n)
        roughness = np.rint(RectBivariateSpline(coarse, coarse, noise)(fine, fine)).astype(np.int16)
        raw = raw + roughness

    heights = np.zeros((count, count), dtype=np.float64)
    heights[border:-border, border:-border] = raw * VERTICAL_SCALE
    lo = int((PATCH_SIZE * 0.5 - 1) / HORIZONTAL_SCALE)
    hi = int((PATCH_SIZE * 0.5 + 1) / HORIZONTAL_SCALE)
    origin_z = float(heights[lo:hi, lo:hi].max())
    return Terrain(name, heights, origin_z, difficulty, seed)


def height_rectangles(heights: np.ndarray):
    """Merge equal adjacent cells into disjoint rectangles (x0, x1, y0, y1, z)."""
    active = {}
    for x, row in enumerate(heights):
        cuts = np.r_[0, np.flatnonzero(np.diff(row)) + 1, len(row)]
        current = {}
        for y0, y1 in zip(cuts[:-1], cuts[1:]):
            key = (int(y0), int(y1), float(row[y0]))
            current[key] = active.pop(key, x)
        for (y0, y1, z), x0 in active.items():
            yield x0, x, y0, y1, z
        active = current
    for (y0, y1, z), x0 in active.items():
        yield x0, len(heights), y0, y1, z


def terrain_scene_xml(robot_xml: Path, terrain: Terrain) -> ET.Element:
    """Compose a terrain scene before compilation."""
    root = ET.parse(robot_xml).getroot()
    root.set("model", f"legbot_{terrain.name}")
    compiler = root.find("compiler")
    compiler.set("meshdir", str((robot_xml.parent / compiler.get("meshdir", ".")).resolve()))
    world = root.find("worldbody")
    ground = world.find("geom[@name='ground']")
    if ground is None:
        raise ValueError("Expected the base robot scene to contain ground.")
    world.remove(ground)
    material = ground.get("material", "ground_material")
    attributes = dict(material=material, contype="1", conaffinity="1", condim="3",
                      friction="1.0 0.005 0.0001")

    if terrain.name == "flat":
        world.append(ground)
    else:
        # Only put the apron OUTSIDE the patch. An infinite z=0 plane would
        # silently fill inverted stairs, slopes and negative obstacle heights.
        for name, pos, size in (
            ("west", "-17 0 -0.1", "12.5 29.5 0.1"),
            ("east", "17 0 -0.1", "12.5 29.5 0.1"),
            ("south", "0 -17 -0.1", "4.5 12.5 0.1"),
            ("north", "0 17 -0.1", "4.5 12.5 0.1"),
        ):
            ET.SubElement(world, "geom", name=f"apron_{name}", type="box", pos=pos, size=size, **attributes)

        if terrain.name in ("stairs_up", "stairs_down", "obstacles"):
            floor = float(terrain.heights.min()) - 0.2
            # Each cell occupies [x,x+0.1] x [y,y+0.1]. The final samples are
            # the patch boundary, hence one fewer cell than height samples.
            for i, (x0, x1, y0, y1, z) in enumerate(height_rectangles(terrain.heights[:-1, :-1])):
                pos = ((x0 + x1) * 0.05 - 4.5, (y0 + y1) * 0.05 - 4.5, (floor + z) / 2)
                size = ((x1 - x0) * 0.05, (y1 - y0) * 0.05, (z - floor) / 2)
                ET.SubElement(world, "geom", name=f"terrain_cell_{i}", type="box",
                              pos=" ".join(map(str, pos)), size=" ".join(map(str, size)), **attributes)
        else:
            low = float(terrain.heights.min())
            span = max(float(np.ptp(terrain.heights)), 0.001)
            ET.SubElement(root.find("asset"), "hfield", name="training_heightfield",
                          nrow=str(terrain.heights.shape[1]), ncol=str(terrain.heights.shape[0]),
                          size=f"4.5 4.5 {span} 0.2")
            ET.SubElement(world, "geom", name="training_surface", type="hfield",
                          hfield="training_heightfield", pos=f"0 0 {low}", **attributes)

    return root


def load_terrain_model(robot_xml: Path, terrain: Terrain) -> mujoco.MjModel:
    """Compose a scene in memory, keeping the robot's physical parameters intact."""
    root = terrain_scene_xml(robot_xml, terrain)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if model.nhfield:
        # MuJoCo stores rows along Y; training arrays index X first.
        low = float(terrain.heights.min())
        span = max(float(np.ptp(terrain.heights)), 0.001)
        model.hfield_data[:] = ((terrain.heights.T - low) / span).ravel()
    return model


# Adjacent 9 m patches form a 36 x 18 m park. The one robot starts on flat.
PARK_LAYOUT = {name: (float(i % 4 * 9), float(i // 4 * 9))
               for i, name in enumerate(TRAINING_TERRAINS)}


def load_terrain_park(robot_xml: Path, difficulty: float = 0.5, seed: int = 0,
                      *, stair_height: float | None = None,
                      terrain_colors: dict[str, tuple[float, float, float, float]] | None = None) -> mujoco.MjModel:
    """Put all eight terrains and one robot into a model, with optional patch colors."""
    terrains = [generate_terrain(name, difficulty, seed, stair_height=stair_height)
                for name in TRAINING_TERRAINS]
    root = terrain_scene_xml(robot_xml, terrains[0])
    root.set("model", "legbot_eight_terrain_park")
    world = root.find("worldbody")
    world.remove(world.find("geom[@name='ground']"))
    fields = {}
    attributes = dict(material="ground_material", contype="1", conaffinity="1",
                      condim="3", friction="1.0 0.005 0.0001")
    for index, terrain in enumerate(terrains):
        tile = terrain_scene_xml(robot_xml, terrain)
        x, y = PARK_LAYOUT[terrain.name]
        patch_material = None
        if terrain_colors is not None and terrain.name in terrain_colors:
            color = " ".join(map(str, terrain_colors[terrain.name]))
            patch_material = f"{terrain.name}_ground_material"
            ET.SubElement(root.find("asset"), "material", name=patch_material, rgba=color,
                          reflectance="0", specular="0.05", shininess="0.1")
        for field in tile.findall("asset/hfield"):
            name = f"{terrain.name}_heightfield"
            field.set("name", name)
            root.find("asset").append(field)
            fields[name] = terrain.heights
        for geom in tile.findall("worldbody/geom"):
            name = geom.get("name", "")
            if name != "ground" and name != "training_surface" and not name.startswith("terrain_cell_"):
                continue  # Do not copy a tile's outside apron into other tiles.
            if name == "ground":
                geom.set("type", "box")
                geom.set("pos", "0 0 -0.1")
                geom.set("size", "4.5 4.5 0.1")
            pos = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
            pos[:2] += [x, y]
            geom.set("pos", " ".join(map(str, pos)))
            geom.set("name", f"{terrain.name}_{name}")
            if patch_material is not None:
                geom.set("material", patch_material)
            if geom.get("hfield"):
                geom.set("hfield", f"{terrain.name}_heightfield")
            world.append(geom)
        ET.SubElement(world, "site", name=f"{index + 1}_{terrain.name}",
                      pos=f"{x - 3} {y + 3.6} 0.15", size="0.025", group="5", rgba="0.05 0.25 0.5 1")

    # Flat approach area surrounds the entire park; no plane under the patches.
    for name, pos, size in (
        ("west", "-9.5 4.5 -0.1", "5 19 0.1"),
        ("east", "36.5 4.5 -0.1", "5 19 0.1"),
        ("south", "13.5 -9.5 -0.1", "18 5 0.1"),
        ("north", "13.5 18.5 -0.1", "18 5 0.1"),
    ):
        ET.SubElement(world, "geom", name=f"park_apron_{name}", type="box", pos=pos, size=size, **attributes)
    ET.SubElement(root, "statistic", center="13.5 4.5 0", extent="22")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    for name, heights in fields.items():
        field = model.hfield(name)
        size = int(field.nrow[0] * field.ncol[0])
        start = int(field.adr[0])
        low = float(heights.min())
        span = max(float(np.ptp(heights)), 0.001)
        model.hfield_data[start:start + size] = ((heights.T - low) / span).ravel()
    return model


def configure_robot_geometry(model: mujoco.MjModel) -> mujoco.MjvOption:
    """Display the existing URDF collision shapes without altering contacts."""
    options = mujoco.MjvOption()
    options.geomgroup[2] = 0
    options.geomgroup[3] = 1
    collisions = (model.geom_bodyid > 0) & (model.geom_contype != 0)
    visuals = (model.geom_bodyid > 0) & (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
    model.geom_rgba[visuals, 3] = 0.0
    model.geom_rgba[collisions, 3] = 1.0
    return options


def configure_scene_theme(model: mujoco.MjModel) -> None:
    """Use solid sage-green terrain without a grid and a pale background."""
    # MuJoCo 3.8 renamed tex_rgb to tex_data and added per-texture channels.
    texture_data = model.tex_data if hasattr(model, "tex_data") else model.tex_rgb
    for i in range(model.ntex):
        name = model.texture(i).name
        if name not in ("ground_checker", "skybox"):
            continue
        width, height = int(model.tex_width[i]), int(model.tex_height[i])
        channels = int(model.tex_nchannel[i]) if hasattr(model, "tex_nchannel") else 3
        start = int(model.tex_adr[i])
        pixels = texture_data[start:start + width * height * channels].reshape(height, width, channels)
        if name == "ground_checker":
            pixels[:, :, :3] = [103, 140, 115]
        else:
            # Six square faces with a soft, pale green gradient.
            t = (np.arange(height) % width) / max(1, width - 1)
            gradient = (1 - t[:, None]) * [218, 229, 215] + t[:, None] * [237, 242, 231]
            pixels[:, :, :3] = gradient[:, None, :].astype(np.uint8)

    for i in range(model.ngeom):
        if model.geom_bodyid[i] == 0 and model.geom_matid[i] < 0:
            model.geom_rgba[i] = [103 / 255, 140 / 255, 115 / 255, 1]
    # Side lighting and shadows make terrain relief readable on a solid surface.
    model.light_dir[:] = [-0.4, -0.6, -1.0]
    model.light_castshadow[:] = True
    model.light_diffuse[:] = 0.65
    model.light_ambient[:] = 0.15
    model.vis.headlight.diffuse[:] = 0.3
    model.vis.headlight.ambient[:] = 0.16
    model.vis.headlight.specular[:] = 0.03
    model.vis.rgba.haze[:] = [0.86, 0.90, 0.84, 1]

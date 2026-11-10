"""C1 CTS sim2sim using the training URDF and the run's resolved configuration.

Left stick: forward / sideways. Right stick: yaw. R or gamepad Y: reset.
Close the viewer to finish. Policy inference and physics run on the CPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

from legbot_terrains import VERTICAL_SCALE, load_terrain_park
from utils import gravity_from_quat, init_joystick, read_joystick_command, setup_tracking_camera

ROOT = Path(__file__).resolve().parents[2]
NAMES = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR")
         for joint in ("hip", "thigh", "calf", "foot")]
WHEELS = np.array([3, 7, 11, 15])
LEGS = np.array([i for i in range(16) if i not in WHEELS])
GROUND_RGBA = (0.82, 0.71, 0.49, 1.0)  # Light sandy brown, approximately #D1B57D.
TERRAIN_RGBA = {
    "flat": (0.88, 0.78, 0.57, 1.0),          # Sand.
    "wave": (0.58, 0.76, 0.70, 1.0),          # Sage green.
    "slope_up": (0.58, 0.70, 0.82, 1.0),      # Soft blue.
    "slope_down": (0.85, 0.65, 0.51, 1.0),    # Peach.
    "rough_slope": (0.70, 0.73, 0.48, 1.0),   # Olive.
    "stairs_up": (0.80, 0.58, 0.64, 1.0),     # Dusty rose.
    "stairs_down": (0.69, 0.62, 0.81, 1.0),   # Lavender.
    "obstacles": (0.76, 0.55, 0.37, 1.0),     # Warm clay.
}


class CTSOnnxPolicy:
    """Adapt the stateless 228-value ONNX student to the 57-value control loop."""

    def __init__(self, path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != "obs" or inputs[0].shape != [1, 228]
                or inputs[0].type != "tensor(float)" or len(outputs) != 1
                or outputs[0].name != "actions" or outputs[0].shape != [1, 16]
                or outputs[0].type != "tensor(float)"):
            raise ValueError("Expected CTS ONNX float32 obs[1,228] -> actions[1,16]")
        self.reset()

    def reset(self):
        self.frames = None

    def __call__(self, single_obs):
        single = single_obs.detach().cpu().numpy().astype(np.float32, copy=False)
        if single.shape != (1, 57):
            raise ValueError(f"Expected a single C1 observation [1,57], got {single.shape}")
        if self.frames is None:
            self.frames = np.repeat(single, 4, axis=0)
        else:
            self.frames = np.concatenate((self.frames[1:], single), axis=0)
        blocks, offset = [], 0
        for width in (3, 3, 3, 16, 16, 16):
            blocks.append(self.frames[:, offset:offset + width].reshape(1, -1))
            offset += width
        history = np.ascontiguousarray(np.concatenate(blocks, axis=1))
        actions = self.session.run(["actions"], {"obs": history})[0]
        return torch.from_numpy(actions)


def load_policy(path):
    """Load either a CTS TorchScript student or its stateless ONNX export."""
    path = Path(path).resolve()
    if path.suffix.lower() == ".onnx":
        return CTSOnnxPolicy(path)
    return torch.jit.load(str(path), map_location="cpu").eval()


class ConfigLoader(yaml.SafeLoader):
    pass


ConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: loader.construct_sequence(node))
ConfigLoader.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice",
                             lambda loader, node: slice(*loader.construct_sequence(node)))


def values(text):
    return np.fromstring(text, sep=" ")


def fmt(array):
    return " ".join(str(float(x)) for x in array)


def origin(element):
    item = element.find("origin")
    if item is None:
        return {"pos": "0 0 0", "euler": "0 0 0"}
    return {"pos": item.get("xyz", "0 0 0"), "euler": item.get("rpy", "0 0 0")}


def visual_mesh_path(path):
    """Losslessly re-encode large binary STLs beyond MuJoCo's STL face limit."""
    if path.suffix.lower() != ".stl":
        return path
    raw = path.read_bytes()
    face_count = int.from_bytes(raw[80:84], "little")
    if face_count <= 200000:
        return path
    if len(raw) != 84 + face_count * 50:
        raise ValueError(f"Expected binary STL: {path}")
    cache = ROOT / "outputs/sim2sim/c1_visual_mesh_cache"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{path.stem}_{hashlib.sha256(raw).hexdigest()[:16]}.obj"
    if not target.exists():
        triangles = np.frombuffer(raw, offset=84, count=face_count,
                                  dtype=[("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
        vertices, indices = np.unique(triangles["vertices"].reshape(-1, 3), axis=0, return_inverse=True)
        temporary = target.with_suffix(".obj.tmp")
        with temporary.open("w") as stream:
            stream.write(f"# Lossless conversion of {path.name}; {face_count} original triangles\n")
            np.savetxt(stream, vertices, fmt="v %.9g %.9g %.9g")
            np.savetxt(stream, indices.reshape(-1, 3) + 1, fmt="f %d %d %d")
        temporary.replace(target)
    return target


def build_model(cfg, terrain, difficulty, *, stair_height=None):
    """Load URDF STL visuals and preserve the separate training collision pieces."""
    urdf = Path(cfg["scene"]["robot"]["spawn"]["asset_path"])
    source = ET.parse(urdf).getroot()
    root = ET.Element("mujoco", model="c1_sim2sim")
    ET.SubElement(root, "compiler", angle="radian", eulerseq="xyz", autolimits="true",
                  meshdir=str(urdf.parent), inertiafromgeom="false", balanceinertia="false")
    ET.SubElement(root, "option", timestep=str(cfg["sim"]["dt"]), integrator="implicitfast",
                  gravity="0 0 -9.81", iterations="100")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="0", armature="0", frictionloss="0")
    ET.SubElement(default, "geom", friction="1 0.005 0.0001", condim="3", group="3")
    assets = ET.SubElement(root, "asset")
    ET.SubElement(assets, "material", name="ground_material", rgba=fmt(GROUND_RGBA),
                  reflectance="0", specular="0.05", shininess="0.1")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 5", dir="-0.4 -0.6 -1", directional="true")
    ET.SubElement(world, "geom", name="ground", type="plane", size="40 40 0.1",
                  material="ground_material")
    links = {link.get("name"): link for link in source.findall("link")}
    joints = source.findall("joint")
    children = {joint.find("child").get("link") for joint in joints}
    base_names = set(links) - children
    if base_names != {"base_link"} or len(joints) != 16:
        raise ValueError("Expected C1 base_link and its 16 actuated joints")
    contact = ET.SubElement(root, "contact")

    def add_body(parent, name, joint=None):
        link = links[name]
        body = ET.SubElement(parent, "body", name=name, **({} if joint is None else origin(joint)))
        if joint is None:
            ET.SubElement(body, "freejoint", name="root")
        else:
            limit = joint.find("limit")
            attrs = dict(name=joint.get("name"), type="hinge", axis=joint.find("axis").get("xyz"))
            lower, upper = float(limit.get("lower", "0")), float(limit.get("upper", "0"))
            if lower < upper:
                attrs["range"] = f"{lower} {upper}"
            else:
                attrs["limited"] = "false"
            ET.SubElement(body, "joint", **attrs)
            ET.SubElement(contact, "exclude", body1=joint.find("parent").get("link"), body2=name)
        inertial = link.find("inertial")
        inertia = inertial.find("inertia")
        pose = origin(inertial)
        if np.any(values(pose["euler"]) != 0):
            raise ValueError("C1 conversion expects body-aligned URDF inertias")
        ET.SubElement(body, "inertial", pos=pose["pos"], mass=inertial.find("mass").get("value"),
                      fullinertia=" ".join(inertia.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")))
        for index, visual in enumerate(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                raise ValueError(f"Expected an STL visual mesh for {name}")
            mesh_name = f"{name}_visual_mesh_{index}"
            ET.SubElement(assets, "mesh", name=mesh_name,
                          file=str(visual_mesh_path((urdf.parent / mesh.get("filename")).resolve())),
                          scale=mesh.get("scale", "1 1 1"))
            color = visual.find("material/color")
            ET.SubElement(body, "geom", name=f"{name}_visual_{index}", type="mesh", mesh=mesh_name,
                          **origin(visual), contype="0", conaffinity="0", group="2", mass="0",
                          rgba=color.get("rgba") if color is not None else "0.8 0.8 0.8 1")
        for index, collision in enumerate(link.findall("collision")):
            geom = collision.find("geometry")[0]
            attrs = dict(name=f"{name}_collision_{index}", **origin(collision))
            if geom.tag == "mesh":
                mesh_name = f"{name}_mesh_{index}"
                ET.SubElement(assets, "mesh", name=mesh_name,
                              file=str((urdf.parent / geom.get("filename")).resolve()),
                              scale=geom.get("scale", "1 1 1"))
                attrs.update(type="mesh", mesh=mesh_name)
            elif geom.tag == "box":
                attrs.update(type="box", size=fmt(values(geom.get("size")) / 2))
            elif geom.tag == "cylinder":
                attrs.update(type="cylinder", size=f"{geom.get('radius')} {float(geom.get('length')) / 2}")
            elif geom.tag == "sphere":
                attrs.update(type="sphere", size=geom.get("radius"))
            else:
                raise ValueError(f"Unsupported collision geometry: {geom.tag}")
            ET.SubElement(body, "geom", **attrs)
        for child_joint in joints:
            if child_joint.find("parent").get("link") == name:
                add_body(body, child_joint.find("child").get("link"), child_joint)

    add_body(world, "base_link")
    actuators = ET.SubElement(root, "actuator")
    for name in NAMES:
        ET.SubElement(actuators, "motor", name=name, joint=name, gear="1")
    xml = ET.tostring(root, encoding="unicode")
    if terrain == "all":
        with tempfile.TemporaryDirectory(prefix="c1_mujoco_") as folder:
            path = Path(folder) / "c1.xml"
            path.write_text(xml)
            model = load_terrain_park(path, difficulty, 0, stair_height=stair_height,
                                      terrain_colors=TERRAIN_RGBA)
    else:
        model = mujoco.MjModel.from_xml_string(xml)
    # Keep the approach area sandy brown and distinguish the eight terrain patches.
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == 0:
            model.geom_group[geom_id] = 0
            name = model.geom(geom_id).name
            color = next((rgba for terrain_name, rgba in TERRAIN_RGBA.items()
                          if terrain == "all" and name.startswith(terrain_name + "_")), GROUND_RGBA)
            model.geom_rgba[geom_id] = color
    assert (model.nq, model.nv, model.nu) == (23, 22, 16)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path, help="CTS student .pt or .onnx file")
    parser.add_argument("--training-config", required=True, type=Path)
    parser.add_argument("--terrain", choices=("flat", "all"), default="flat")
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--stair-height", type=float, default=0.22,
                        help="Fixed stair riser height in metres for --terrain all (default: 0.22)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--cmd", nargs=3, type=float)
    parser.add_argument("--max-cmd", nargs=3, type=float, default=[0.5, 0.5, 0.785])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be finite and positive")
    if not np.isfinite(args.difficulty) or not 0 <= args.difficulty <= 1:
        parser.error("--difficulty must be between zero and one")
    if not np.isfinite(args.stair_height) or args.stair_height < VERTICAL_SCALE:
        parser.error(f"--stair-height must be finite and at least {VERTICAL_SCALE} m")
    if not np.isfinite(args.max_cmd).all() or np.any(np.array(args.max_cmd) < 0):
        parser.error("--max-cmd must contain finite nonnegative values")
    if args.cmd is not None and not np.isfinite(args.cmd).all():
        parser.error("--cmd must contain finite values")
    torch.set_num_threads(1)
    cfg = yaml.load(args.training_config.read_text(), Loader=ConfigLoader)
    model = build_model(cfg, args.terrain, args.difficulty, stair_height=args.stair_height)
    data = mujoco.MjData(model)
    qids = np.array([int(model.jnt_qposadr[model.joint(n).id]) for n in NAMES])
    vids = np.array([int(model.jnt_dofadr[model.joint(n).id]) for n in NAMES])
    initial = cfg["scene"]["robot"]["init_state"]
    defaults = np.array([next(v for expr, v in initial["joint_pos"].items() if re.fullmatch(expr, n)) for n in NAMES])
    groups = cfg["scene"]["robot"]["actuators"]
    kp, kd = np.zeros(16), np.zeros(16)
    for indices, key in ((LEGS, "legs"), (WHEELS, "wheels")):
        kp[indices], kd[indices] = groups[key]["stiffness"], groups[key]["damping"]
    policy = load_policy(args.policy)
    actions = np.zeros(16, dtype=np.float32)
    target_pos, target_vel = defaults.copy(), np.zeros(16)
    obs_cfg = cfg["observations"]["policy"]

    def reset():
        mujoco.mj_resetData(model, data)
        data.qpos[:3], data.qpos[3:7] = initial["pos"], initial["rot"]
        data.qpos[qids] = defaults
        actions[:] = 0
        target_pos[:], target_vel[:] = defaults, 0
        policy.reset()
        mujoco.mj_forward(model, data)

    def obs(command):
        pos = data.qpos[qids] - defaults
        pos[WHEELS] = 0
        terms = [data.qvel[3:6], gravity_from_quat(data.qpos[3:7]), command,
                 pos, data.qvel[vids], actions]
        names = ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions")
        return np.concatenate([np.clip(value, *obs_cfg[name]["clip"]) * obs_cfg[name]["scale"]
                               for name, value in zip(names, terms)]).astype(np.float32)

    reset()
    joystick = None if args.headless or args.cmd is not None else init_joystick()
    if not args.headless and args.cmd is None and joystick is None:
        raise RuntimeError("No gamepad detected; connect it before starting the manual test")
    gamepad_name = joystick.get_name() if joystick is not None else None
    command = np.array(args.cmd if args.cmd is not None else [0, 0, 0], dtype=np.float32)
    pending_reset = [False]
    def key_callback(key):
        if key == ord("R"):
            pending_reset[0] = True

    duration = args.duration if args.duration is not None else (20 if args.headless else float("inf"))
    viewer = None if args.headless else mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
    base_id = model.body("base_link").id
    if viewer is not None:
        setup_tracking_camera(viewer, trackbodyid=base_id, distance=2.6, elevation=-25.0, azimuth=135.0)
        viewer.opt.geomgroup[2] = 1
        viewer.opt.geomgroup[3] = 0
    print(f"C1 ready | policy={args.policy} | terrain={args.terrain} | mass={model.body_mass.sum():.6f} kg", flush=True)
    print("Left stick: forward / sideways; right stick: yaw. R / Y: reset. Close window to finish.", flush=True)
    wall_start = time.monotonic()
    steps, falls, resets = 0, 0, 0
    was_fallen = False
    previous_y = False
    min_upright, max_distance = 1.0, 0.0
    velocity_sum = np.zeros(3)
    velocity_samples = 0
    command_samples = []
    decimation = cfg["decimation"]
    warnings = [mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL]
    status = "completed"
    try:
        with torch.inference_mode():
            while steps * model.opt.timestep < duration and (viewer is None or viewer.is_running()):
                step_start = time.monotonic()
                if steps % decimation == 0:
                    if joystick is not None:
                        command[:] = read_joystick_command(joystick, np.array(args.max_cmd))
                        y = bool(joystick.get_button(3))
                        pending_reset[0] |= y and not previous_y
                        previous_y = y
                    if pending_reset[0]:
                        reset()
                        resets += 1
                        pending_reset[0] = False
                        was_fallen = False
                    single = obs(command)
                    result = policy(torch.from_numpy(single).unsqueeze(0)).numpy().reshape(16)
                    if not np.isfinite(result).all():
                        status = "unstable"
                        break
                    actions[:] = np.clip(result, -10, 10)
                    target_pos[LEGS] = defaults[LEGS] + actions[LEGS] * 0.25
                    target_vel[WHEELS] = np.clip(actions[WHEELS] * 20, -150, 150)
                    rotation = data.xmat[base_id].reshape(3, 3)
                    velocity_sum += rotation.T @ data.qvel[:3]
                    velocity_samples += 1
                    if steps % (decimation * 5) == 0:
                        command_samples.append([steps * model.opt.timestep, *command.tolist()])
                torque = kp * (target_pos - data.qpos[qids]) + kd * (target_vel - data.qvel[vids])
                for indices, group in ((LEGS, groups["legs"]), (WHEELS, groups["wheels"])):
                    curve = np.array(group["torque_speed_curve"])
                    cap = np.minimum(np.interp(np.abs(data.qvel[vids[indices]]), curve[:, 0], curve[:, 1]),
                                     group["effort_limit"])
                    torque[indices] = np.clip(torque[indices], -cap, cap)
                data.ctrl[:] = torque
                mujoco.mj_step(model, data)
                steps += 1
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or any(data.warning[w].number for w in warnings):
                    status = "unstable"
                    break
                upright = -float(gravity_from_quat(data.qpos[3:7])[2])
                min_upright = min(min_upright, upright)
                fallen = upright < 0.5
                if fallen and not was_fallen:
                    falls += 1
                    print(f"Fall detected at {steps * model.opt.timestep:.2f}s; R / Y resets the robot", flush=True)
                was_fallen = fallen
                max_distance = max(max_distance, float(np.linalg.norm(data.qpos[:2])))
                if viewer is not None:
                    if steps % 4 == 0:
                        viewer.sync()
                    remaining = model.opt.timestep - (time.monotonic() - step_start)
                    if remaining > 0:
                        time.sleep(remaining)
    finally:
        if viewer is not None:
            viewer.close()
        if joystick is not None:
            import pygame
            pygame.quit()
    report = dict(status=status if status == "unstable" else "fallen" if falls else "completed",
                  stair_height_m=(round(args.stair_height / VERTICAL_SCALE) * VERTICAL_SCALE
                                  if args.terrain == "all" else None),
                  policy=str(args.policy.resolve()), training_config=str(args.training_config.resolve()),
                  terrain=args.terrain, sim_seconds=steps * model.opt.timestep,
                  wall_seconds=time.monotonic() - wall_start, steps=steps, falls=falls, resets=resets,
                  min_upright_cos=min_upright, max_distance_m=max_distance, final_xyz=data.qpos[:3].tolist(),
                  mean_base_velocity=(velocity_sum / max(velocity_samples, 1)).tolist(),
                  gamepad=gamepad_name,
                  control_source="gamepad" if joystick is not None else "fixed", command_samples=command_samples,
                  geometry="urdf_stl_visuals", colors="urdf_materials", collision_geometry="training_urdf",
                  numerical_warnings={w.name: int(data.warning[w].number) for w in warnings})
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "command_samples"}), flush=True)
    return int(args.headless and (falls > 0 or status == "unstable"))


if __name__ == "__main__":
    raise SystemExit(main())

"""Geometry checks bypassing both adjacency exclusions and same-body filtering."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "resources/w1w_wf"
AVAILABLE = importlib.util.find_spec("mujoco") is not None and (ROOT / "legbot.xml").exists()


def box_separation(model, data, a, b):
    """OBB separating axis test; positive means separated, negative means overlap."""
    ra, rb = data.geom_xmat[a].reshape(3, 3), data.geom_xmat[b].reshape(3, 3)
    axes = [*ra.T, *rb.T, *[np.cross(x, y) for x in ra.T for y in rb.T]]
    delta = data.geom_xpos[b] - data.geom_xpos[a]
    gaps = []
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            continue
        axis = axis/norm
        reach = np.abs(axis @ ra) @ model.geom_size[a] + np.abs(axis @ rb) @ model.geom_size[b]
        gaps.append(abs(delta @ axis)-reach)
    return max(gaps)


@unittest.skipUnless(AVAILABLE, "Requires MuJoCo and local model")
class CollisionGeometryTests(unittest.TestCase):
    def test_reset_wheels_start_above_flat_ground(self):
        import itertools
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(ROOT / "legbot.xml"))
        data = mujoco.MjData(model)
        stand = np.array([0,-.68,1.4,0, 0,.68,-1.4,0] * 2)
        # Extremes of independent thigh/calf reset scales on every leg.
        for scales in itertools.product((.9, 1.1), repeat=8):
            pose = stand.copy()
            pose[[1,2,5,6,9,10,13,14]] *= scales
            data.qpos[:7] = [0,0,.53,1,0,0,0]
            data.qpos[7:] = pose
            mujoco.mj_forward(model, data)
            for i in range(model.ngeom):
                if model.geom(i).name.endswith("foot_collision"):
                    self.assertGreater(data.geom_xpos[i,2] - model.geom_size[i,0], .005)

    def test_all_pairs_in_reference_and_nearby_poses(self):
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(ROOT / "legbot.xml"))
        data = mujoco.MjData(model)
        ids = [i for i in range(model.ngeom) if "_collision" in model.geom(i).name]
        stand = np.array([0,-.68,1.4,0, 0,.68,-1.4,0, 0,-.68,1.4,0, 0,.68,-1.4,0])
        rng = np.random.default_rng(42)
        poses = [stand, np.zeros(16)] + [stand + rng.uniform(-1,1,16)*np.tile([.15,.25,.25,np.pi],4) for _ in range(32)]
        for index, pose in enumerate(poses):
            data.qpos[:7] = [0,0,.53,1,0,0,0]
            data.qpos[7:] = pose
            mujoco.mj_forward(model, data)
            for k, a in enumerate(ids):
                for b in ids[k+1:]:
                    if model.geom_type[a] == model.geom_type[b] == mujoco.mjtGeom.mjGEOM_BOX:
                        distance = box_separation(model, data, a, b)
                    else:
                        distance = mujoco.mj_geomDistance(model, data, a, b, .05, None)
                    self.assertGreaterEqual(distance, -1e-5, (index, model.geom(a).name, model.geom(b).name, distance))


if __name__ == "__main__":
    unittest.main()

"""USD tests without launching Isaac Sim (requires the bundled pxr libraries)."""
import importlib.util
from pathlib import Path
import unittest

try:
    from pxr import Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None


@unittest.skipIf(Usd is None, "USD runtime not on Python/library paths")
class AdjacentFilterTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "source/robot_lab/robot_lab/assets/collision_filters.py"
        spec = importlib.util.spec_from_file_location("collision_filters", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.apply = module.filter_adjacent_links
        self.stage = Usd.Stage.CreateInMemory()
        self.robot = UsdGeom.Xform.Define(self.stage, "/env0/robot").GetPrim()
        for name in ["base", "hip", "thigh"]:
            prim = UsdGeom.Xform.Define(self.stage, "/env0/robot/" + name).GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
        for name, a, b in [("hip_joint", "base", "hip"), ("thigh_joint", "hip", "thigh")]:
            joint = UsdPhysics.RevoluteJoint.Define(self.stage, "/env0/robot/" + name)
            joint.CreateBody0Rel().SetTargets(["/env0/robot/" + a])
            joint.CreateBody1Rel().SetTargets(["/env0/robot/" + b])
            joint.CreateCollisionEnabledAttr().Set(True)

    def test_only_adjacent_pairs_are_filtered(self):
        self.assertEqual(self.apply(self.robot, ["hip_joint", "thigh_joint"]), 2)
        for name in ["hip_joint", "thigh_joint"]:
            self.assertFalse(UsdPhysics.Joint(self.stage.GetPrimAtPath("/env0/robot/"+name)).GetCollisionEnabledAttr().Get())
        targets = UsdPhysics.FilteredPairsAPI(self.stage.GetPrimAtPath("/env0/robot/base")).GetFilteredPairsRel().GetTargets()
        self.assertEqual([str(p) for p in targets], ["/env0/robot/hip"])
        # Base/thigh and ground are deliberately not excluded.
        self.assertFalse(self.stage.GetPrimAtPath("/env0/robot/thigh").HasAPI(UsdPhysics.FilteredPairsAPI))

    def test_referenced_environment_remaps_targets(self):
        self.apply(self.robot, ["hip_joint", "thigh_joint"])
        clone = UsdGeom.Xform.Define(self.stage, "/env1/robot").GetPrim()
        clone.GetReferences().AddInternalReference("/env0/robot")
        targets = UsdPhysics.FilteredPairsAPI(self.stage.GetPrimAtPath("/env1/robot/base")).GetFilteredPairsRel().GetTargets()
        self.assertEqual([str(p) for p in targets], ["/env1/robot/hip"])

    def test_missing_joint_fails(self):
        with self.assertRaisesRegex(RuntimeError, "Missing collision-filter joints"):
            self.apply(self.robot, ["hip_joint", "missing_joint"])
        self.assertTrue(UsdPhysics.Joint(self.stage.GetPrimAtPath("/env0/robot/hip_joint")).GetCollisionEnabledAttr().Get())


if __name__ == "__main__":
    unittest.main()

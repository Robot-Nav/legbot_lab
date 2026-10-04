"""Validate local W1W assets when available (private assets are not in Git)."""
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "resources" / "w1w_wf"


def vec(text):
    return np.fromstring(text, sep=" ")


@unittest.skipUnless((ROOT / "urdf/w1w_wf.urdf").exists(), "Local W1W assets are unavailable")
class W1WAssetTests(unittest.TestCase):
    def setUp(self):
        self.urdf = ET.parse(ROOT / "urdf/w1w_wf.urdf").getroot()
        self.xml = ET.parse(ROOT / "legbot.xml").getroot()

    def test_original_physics_and_joint_interface(self):
        original = ET.parse(ROOT / "urdf/w1w_urdf1.urdf").getroot()
        links = {l.get("name"): l for l in self.urdf.findall("link")}
        joints = self.urdf.findall("joint")
        expected = [f"{leg}_{part}_joint" for leg in ["fl", "fr", "rl", "rr"]
                    for part in ["hip", "thigh", "calf", "foot"]]
        self.assertEqual([j.get("name") for j in joints], expected)
        self.assertEqual(set(links) - {j.find("child").get("link") for j in joints}, {"base"})
        self.assertEqual(len(links), 17)
        for joint in joints:
            original_name = joint.get("name")
            source_joint = original.find(f'joint[@name="{original_name}"]')
            for tag in ["origin", "axis", "parent", "child"]:
                self.assertEqual(joint.find(tag).attrib, source_joint.find(tag).attrib)
            self.assertAlmostEqual(float(np.linalg.norm(vec(joint.find("axis").get("xyz")))), 1.)
            if joint.get("name").endswith("_foot_joint"):
                self.assertEqual(joint.get("type"), "continuous")
                self.assertNotIn("lower", joint.find("limit").attrib)
            else:
                limit = joint.find("limit")
                self.assertLess(float(limit.get("lower")), float(limit.get("upper")))
        for old in original.findall("link"):
            link = links[old.get("name")]
            for tag in ["inertial", "visual"]:
                a, b = list(old.find(tag).iter()), list(link.find(tag).iter())
                self.assertEqual([(x.tag, x.attrib) for x in a], [(x.tag, x.attrib) for x in b])
            for mesh in link.findall(".//mesh"):
                self.assertTrue((ROOT / "urdf" / mesh.get("filename")).is_file())
            inertia = link.find("inertial/inertia").attrib
            matrix = np.array([[float(inertia["i" + a + b if a <= b else "i" + b + a])
                                for b in "xyz"] for a in "xyz"])
            eig = np.linalg.eigvalsh(matrix)
            self.assertGreater(eig[0], 0.)
            self.assertLessEqual(eig[-1], sum(eig[:2]))

    def test_urdf_mujoco_parity(self):
        for joint in self.urdf.findall("joint"):
            name = joint.get("name")
            mj = self.xml.find(f'.//joint[@name="{name}"]')
            body = self.xml.find(f'.//body[@name="{joint.find("child").get("link")}"]')
            np.testing.assert_allclose(vec(joint.find("origin").get("xyz")), vec(body.get("pos")))
            np.testing.assert_allclose(vec(joint.find("axis").get("xyz")), vec(mj.get("axis")))
            if joint.get("type") == "revolute":
                lim = joint.find("limit")
                np.testing.assert_allclose([float(lim.get("lower")), float(lim.get("upper"))], vec(mj.get("range")))
        for link in self.urdf.findall("link"):
            name = link.get("name")
            body = self.xml.find(f'.//body[@name="{name}"]')
            inertial = link.find("inertial")
            self.assertAlmostEqual(float(inertial.find("mass").get("value")), float(body.find("inertial").get("mass")))
            np.testing.assert_allclose(vec(inertial.find("origin").get("xyz")), vec(body.find("inertial").get("pos")), atol=1e-9)
            values = inertial.find("inertia").attrib
            np.testing.assert_allclose([float(values[key]) for key in ["ixx", "iyy", "izz", "ixy", "ixz", "iyz"]],
                                       vec(body.find("inertial").get("fullinertia")), atol=1e-9)
            collisions = link.findall("collision")
            mj_collisions = [g for g in body.findall("geom") if g.get("class") == "collision"]
            self.assertEqual(len(collisions), len(mj_collisions))
            for col in collisions:
                geom = body.find(f'geom[@name="{col.get("name", name + "_collision")}"]')
                self.assertIsNotNone(geom)
                np.testing.assert_allclose(vec(col.find("origin").get("xyz")), vec(geom.get("pos")))
                np.testing.assert_allclose(vec(col.find("origin").get("rpy")), vec(geom.get("euler", "0 0 0")))
                shape = list(col.find("geometry"))[0]
                self.assertEqual(shape.tag, geom.get("type"))
                sizes = vec(shape.get("size"))/2 if shape.tag == "box" else [float(shape.get("radius")), float(shape.get("length"))/2]
                np.testing.assert_allclose(sizes, vec(geom.get("size")))

    def test_explicit_adjacent_exclusions(self):
        expected = {frozenset([j.find("parent").get("link"), j.find("child").get("link")])
                    for j in self.urdf.findall("joint")}
        actual = {frozenset([e.get("body1"), e.get("body2")]) for e in self.xml.findall("contact/exclude")}
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 16)

    def test_adjacent_calf_wheel_clearance(self):
        # Projection on the wheel axis is a separating axis, independent of wheel angle.
        for leg in ["fl", "fr", "rl", "rr"]:
            calves = self.urdf.findall(f'link[@name="{leg}_calf"]/collision')
            wheel = self.urdf.find(f'link[@name="{leg}_foot"]/collision')
            joint = self.urdf.find(f'joint[@name="{leg}_foot_joint"]')
            wheel_y = vec(joint.find("origin").get("xyz"))[1] + vec(wheel.find("origin").get("xyz"))[1]
            for calf in calves:
                center_y = vec(calf.find("origin").get("xyz"))[1]
                shape = list(calf.find("geometry"))[0]
                rpy = vec(calf.find("origin").get("rpy"))
                if shape.tag == "box":
                    # All current boxes rotate only about y: their y thickness is invariant.
                    np.testing.assert_allclose(rpy[[0, 2]], [0, 0])
                    half_width = vec(shape.get("size"))[1]/2
                else:
                    np.testing.assert_allclose(rpy, [np.pi/2, 0, 0])
                    half_width = float(shape.get("length"))/2
                gap = abs(wheel_y-center_y) - half_width - float(wheel.find("geometry/cylinder").get("length"))/2
                self.assertGreaterEqual(gap, .01099)



if __name__ == "__main__":
    unittest.main()

"""Explicit adjacent-link collision exclusions, authored before environment cloning."""
from pxr import Usd, UsdPhysics


def filter_adjacent_links(robot_prim, joint_names):
    """Disable only connected body pairs; preserve terrain and non-adjacent contacts.

    Fail on missing joints/targets rather than silently training with incomplete filters.
    """
    stage = robot_prim.GetStage()
    root = robot_prim.GetPath()
    wanted = set(joint_names)
    joints = {}
    for prim in Usd.PrimRange(robot_prim):
        if prim.IsA(UsdPhysics.Joint) and prim.GetName() in wanted:
            if prim.GetName() in joints:
                raise RuntimeError(f"Duplicate joint name: {prim.GetName()}")
            joints[prim.GetName()] = UsdPhysics.Joint(prim)
    missing = wanted - joints.keys()
    if missing:
        raise RuntimeError(f"Missing collision-filter joints: {sorted(missing)}")
    pairs = []
    for name in sorted(wanted):
        joint = joints[name]
        parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if len(parents) != 1 or len(children) != 1:
            raise RuntimeError(f"Expected two connected bodies for joint {name}")
        parent, child = parents[0], children[0]
        if parent == child or not parent.HasPrefix(root) or not child.HasPrefix(root):
            raise RuntimeError(f"Invalid connected body paths for joint {name}: {parent}, {child}")
        for path in (parent, child):
            body = stage.GetPrimAtPath(path)
            if not body.IsValid() or not body.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"Joint {name} does not target a rigid body: {path}")
        pairs.append((joint, parent, child))
    for joint, parent, child in pairs:
        joint.CreateCollisionEnabledAttr().Set(False)
        # Body-level pair filtering covers every collision shape on both links.
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(parent)).CreateFilteredPairsRel().AddTarget(child)
    return len(pairs)

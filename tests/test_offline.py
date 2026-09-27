import ast
import importlib.util
import tempfile
import types
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from manipulation.kinematics import forward_kinematics, inverse_kinematics
from manipulation.sensor_contracts import depth_meters, validate_snapshot

ROOT = Path(__file__).resolve().parents[1]


def method(file, name, namespace=None):
    tree = ast.parse((ROOT / file).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {"np": np, **(namespace or {})}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(file), "exec"), scope)
    return scope[name]


class OfflineTests(unittest.TestCase):
    def test_uint16_padded_and_endian(self):
        raw=np.array([[1000,2000,999],[3000,4000,999]],dtype=">u2").tobytes()
        np.testing.assert_allclose(depth_meters(raw,2,2,6,"16UC1",True),[[1,2],[3,4]])

    def test_float32_depth_is_meters(self):
        raw=np.ones((2,2),dtype="<f4").tobytes()
        np.testing.assert_allclose(depth_meters(raw,2,2,8,"32FC1"),np.ones((2,2)))

    def test_invalid_depth_contract(self):
        with self.assertRaises(ValueError):depth_meters(b"",2,2,8,"32FC1")
        with self.assertRaises(ValueError):depth_meters(b"1234",1,2,4,"8UC1")

    def test_snapshot_frame_and_time(self):
        header=NS(frame_id="camera",stamp=NS(sec=10,nanosec=0))
        depth=NS(header=header,height=2,width=2)
        info=NS(header=header,height=2,width=2,k=[1,0,0,0,1,0,0,0,1],d=[])
        validate_snapshot(np.zeros((2,2,3)),depth,info,header,10.1)
        with self.assertRaises(ValueError):validate_snapshot(np.zeros((2,2,3)),depth,info,header,11)
        depth.header=NS(frame_id="other",stamp=header.stamp)
        with self.assertRaises(ValueError):validate_snapshot(np.zeros((2,2,3)),depth,info,header,10.1)

    def test_fk_ik_roundtrip(self):
        q=np.array([-0.4,0.4,0.9,1.2,0.0])
        target=forward_kinematics(q)
        solution=inverse_kinematics(q+0.01,target)
        self.assertIsNotNone(solution)
        self.assertLess(solution["pos_error"],1e-5)
        self.assertLess(solution["rotation_error"],1e-5)

    def test_unreachable_target_rejected(self):
        target=np.eye(4);target[0,3]=10
        self.assertIsNone(inverse_kinematics(np.zeros(5),target))

    def test_nan_workspace_rejected(self):
        check=method("manipulation/grasping.py","_workspace_ok",{"math":__import__("math")})
        self.assertFalse(check(NS(),np.array([np.nan]*3)))

    def test_out_of_bounds_not_clipped(self):
        check=method("manipulation/grasping.py","_check_joint_limits")
        stub=NS(joint_limits_lower=np.full(5,-1),joint_limits_upper=np.full(5,1))
        self.assertFalse(check(stub,np.array([0,0,1.1,0,0]))[0])

    def test_failed_navigation_does_not_update_zone(self):
        run=method("final_project_runner.py","run_phase4_execution")
        logger=NS(info=lambda *a:None,warn=lambda *a:None,error=lambda *a:None)
        state=NS(plan_steps=[{"action":"go","zone_image":"dog"},{"action":"pick"}],zone_id_by_zone_image={"dog":2},current_zone_id=1,get_logger=lambda:logger,get_zone_pose_from_id=lambda i:(0,0,0),navigate_to=lambda *a:False,pick_cube=lambda:self.fail("pick after failed navigation"))
        self.assertFalse(run(state))
        self.assertEqual(state.current_zone_id,1)

    def test_action_file_absence_fails(self):
        fake=types.ModuleType("servo_controller_msgs.msg")
        fake.ServosPosition=fake.ServoPosition=object
        sys.modules["servo_controller_msgs"]=types.ModuleType("servo_controller_msgs")
        sys.modules["servo_controller_msgs.msg"]=fake
        from manipulation.utils.action_group_controller import ActionGroupController
        with tempfile.TemporaryDirectory() as d:
            controller=ActionGroupController(NS(),d)
            with self.assertRaises(FileNotFoundError):controller.run_action("missing")
            self.assertFalse(controller.running_action)


if __name__=="__main__":unittest.main()

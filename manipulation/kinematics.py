"""FK and bounded numerical IK in base_footprint coordinates."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .utils.kinematics_utils import get_urdf

LOWER=np.array([-2.0,-1.65,-1.75,-1.80,-2.0])
UPPER=np.array([2.0,1.65,1.75,1.80,2.0])

def forward_kinematics(q,target="tcp"):
    q=np.asarray(q,dtype=float)
    if q.shape!=(5,) or not np.all(np.isfinite(q)):
        raise ValueError("Expected five finite joint angles")
    geometry=get_urdf()
    pose=geometry["T_base"].copy()
    for i in range(1,6):
        static=geometry["T_base_joint_1" if i==1 else f"T_joint{i-1}_joint_{i}"]
        dynamic=np.eye(4)
        dynamic[:3,:3]=Rotation.from_rotvec(geometry[f"Axis_joint_{i}"]*q[i-1]).as_matrix()
        pose=pose@static@dynamic
        if target==f"j{i}":return pose
        if i==4 and target=="cam":return pose@geometry["T_joint4_cam_offset"]
    if target!="tcp":raise ValueError("Unknown target frame")
    return pose@geometry["T_joint5_TCP_offset"]

def inverse_kinematics(q,T_target,lower=LOWER,upper=UPPER,max_position_error=0.01,max_rotation_error=0.15):
    target=np.asarray(T_target,dtype=float)
    q=np.asarray(q,dtype=float)
    if target.shape!=(4,4) or not np.all(np.isfinite(target)) or not np.all(np.isfinite(q)):
        return None
    def residual(joints):
        current=forward_kinematics(joints)
        rotation=Rotation.from_matrix(current[:3,:3].T@target[:3,:3]).as_rotvec()
        return np.r_[current[:3,3]-target[:3,3],rotation]
    result=least_squares(residual,np.clip(q,lower,upper),bounds=(lower,upper),method="trf")
    error=residual(result.x)
    pos=float(np.linalg.norm(error[:3])); rot=float(np.linalg.norm(error[3:]))
    if not result.success or not np.all(np.isfinite(result.x)) or pos>max_position_error or rot>max_rotation_error:return None
    return {"sol":result.x,"pos_error":pos,"rotation_error":rot}

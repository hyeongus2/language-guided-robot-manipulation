import os
from .sensor_contracts import depth_meters, validate_snapshot
import time
import math
import cv2
import threading
import rclpy
import numpy as np

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import tf2_ros

from .kinematics import *
from .utils.kinematics_utils import *
from .utils.grasping_base import GraspingNodeBase


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Converts quaternion to 3x3 rotation matrix (right-handed)."""
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    return R


def _make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Create 4x4 homogeneous transform from rotation and translation."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3,)
    return T


class GraspingNode(GraspingNodeBase):
    def __init__(self, name: str):
        super().__init__(name)

        # --- CV / Depth ---
        self.bridge = CvBridge()
        self._data_lock = threading.Lock()
        self._rgb_image = None
        self._depth_msg = None
        self._cam_info = None
        self._rgb_header = None
        self._snapshot = None

        self.rgb_sub = self.create_subscription(
            Image, os.environ.get("ROBOT_RGB_TOPIC", "/depth_cam/rgb/image_rect_color"), self._rgb_cb, 1
        )
        self.depth_sub = self.create_subscription(
            Image, os.environ.get("ROBOT_DEPTH_TOPIC", "/depth_cam/aligned_depth_to_color/image_raw"), self._depth_cb, 1
        )
        self.cam_info_sub = self.create_subscription(
            CameraInfo, os.environ.get("ROBOT_INFO_TOPIC", "/depth_cam/rgb/camera_info"), self._cam_info_cb, 1
        )

        # --- TF ---
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.camera_frame = "depth_cam_color_optical_frame"
        self.world_frame_candidates = ["base_footprint"]

        # --- Safety: conservative joint limits (radians) ---
        # NOTE: Keep conservative limits to avoid awkward/wrist-flip postures.
        self.joint_limits_lower = np.array([-2.0, -1.65, -1.75, -1.80, -2.0], dtype=np.float64)
        self.joint_limits_upper = np.array([ 2.0,  1.65,  1.75,  1.80,  2.0], dtype=np.float64)

        # --- Safety: workspace gating ---
        self.MIN_Z = 0.02       # meters: do not command below this
        self.MAX_Z = 0.35       # meters: reject TF spikes
        self.MIN_R = 0.14       # meters: too close to base is risky
        self.MAX_R = 0.45       # meters: too far is often unreachable

        # --- Safety: posture gating ---
        self.MAX_BASE_YAW_DIFF = 0.52   # rad (~30deg), avoid reaching backward
        self.MAX_WRIST_FLIP = 2.2       # rad, reject extreme wrist rotation (heuristic)
        self.MIN_SHOULDER_PITCH = -1.4  # rad, avoid weird folding
        self.MAX_SHOULDER_PITCH =  1.4

        # --- Motion offsets / timing ---
        self.GRASP_DZ = 0.03     # final z offset above detected depth point
        self.DUR_APPROACH = 3.0  # slow = safer

        self.diagnostic_mode = True

    # ============================================================
    # Data accessors
    # ============================================================

    @property
    def rgb_image(self):
        with self._data_lock:
            return self._rgb_image.copy() if self._rgb_image is not None else None

    @property
    def depth_msg(self):
        with self._data_lock:
            return self._depth_msg

    @property
    def cam_info(self):
        with self._data_lock:
            return self._cam_info

    def _has_camera_data(self) -> bool:
        with self._data_lock:
            return (
                self._rgb_image is not None
                and self._depth_msg is not None
                and self._cam_info is not None
            )

    def _diag(self, msg: str):
        if self.diagnostic_mode:
            self.get_logger().info(f"[DIAG] {msg}")

    def _diag_warn(self, msg: str):
        if self.diagnostic_mode:
            self.get_logger().warn(f"[DIAG] {msg}")

    def _diag_error(self, msg: str):
        if self.diagnostic_mode:
            self.get_logger().error(f"[DIAG] {msg}")

    # ============================================================
    # Joint helpers
    # ============================================================

    def _normalize_angle(self, angle: float) -> float:
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _clamp_joints(self, q: np.ndarray) -> np.ndarray:
        q_clamped = q.copy()
        for i in range(len(q)):
            q_clamped[i] = self._normalize_angle(q_clamped[i])
            q_clamped[i] = np.clip(
                q_clamped[i], self.joint_limits_lower[i], self.joint_limits_upper[i]
            )
        return q_clamped

    def _check_joint_limits(self, q: np.ndarray) -> tuple:
        violations = []
        if np.shape(q) != (5,) or not np.all(np.isfinite(q)):
            return False, ["Non-finite or malformed joints"]
        for i in range(len(q)):
            if q[i] < self.joint_limits_lower[i]:
                violations.append(f"Joint {i+1} below limit")
            elif q[i] > self.joint_limits_upper[i]:
                violations.append(f"Joint {i+1} above limit")
        return len(violations) == 0, violations

    def _check_solution_safety(self, q_sol: np.ndarray, target_yaw: float) -> bool:
        """Heuristic safety checks to reject awkward / risky IK solutions."""
        ok, violations = self._check_joint_limits(q_sol)
        if not ok:
            self._diag_error("Joint limit violation: " + " | ".join(violations))
            return False

        # Avoid reaching backward (large base yaw mismatch)
        base_angle = -float(q_sol[0])
        diff = base_angle - float(target_yaw)
        while diff > np.pi:
            diff -= 2.0 * np.pi
        while diff < -np.pi:
            diff += 2.0 * np.pi
        if abs(diff) > self.MAX_BASE_YAW_DIFF:
            self._diag_error(
                f"[Safety] Base yaw diff too large (base={base_angle:.2f}, target={target_yaw:.2f})"
            )
            return False

        # Avoid extreme wrist rotation (wrist flip heuristic)
        wrist = float(q_sol[4])
        if abs(wrist) > self.MAX_WRIST_FLIP:
            self._diag_error(f"[Safety] Wrist rotation too large: q5={wrist:.2f}")
            return False

        # Avoid extreme shoulder pitch folding
        shoulder = float(q_sol[1])
        if shoulder < self.MIN_SHOULDER_PITCH or shoulder > self.MAX_SHOULDER_PITCH:
            self._diag_error(f"[Safety] Shoulder pitch out of safe range: q2={shoulder:.2f}")
            return False

        return True

    def gripper_close(self, duration=1.2):
        self._set_position_pulse([(10, 550)], duration)

    def gripper_open(self, duration=1.0):
        self._set_position_pulse([(10, 100)], duration)

    def get_joint_positions(self):
        q = self.get_joint_positions_pulse()
        return pulse2angle(q)

    def set_joint_positions(self, q, duration):
        """Normal arm: send IK joint values directly."""
        pulse = angle2pulse(q)
        self.set_joint_positions_pulse(pulse, duration)

    # ============================================================
    # ROS callbacks
    # ============================================================

    def _rgb_cb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self._data_lock:
                self._rgb_image = img
                self._rgb_header = msg.header
        except Exception as e:
            self.get_logger().error(f"[RGB] imgmsg_to_cv2 failed: {e}")

    def _depth_cb(self, msg: Image):
        with self._data_lock:
            self._depth_msg = msg

    def _cam_info_cb(self, msg: CameraInfo):
        with self._data_lock:
            self._cam_info = msg

    # ============================================================
    # Depth / TF utils
    # ============================================================

    def _wait_for_rgb_depth_caminfo(self, timeout_sec=2.0):
        if os.environ.get("ROBOT_ALIGNED_RGB_DEPTH") != "1":
            self.get_logger().error("Confirm rectified/aligned RGB-depth setup before enabling markerless mode")
            return False
        deadline=time.monotonic()+timeout_sec
        while time.monotonic()<deadline:
            try:
                with self._data_lock:
                    rgb=self._rgb_image.copy() if self._rgb_image is not None else None
                    depth=self._depth_msg;info=self._cam_info;header=self._rgb_header
                if rgb is not None and depth is not None and info is not None and header is not None:
                    now=self.get_clock().now().nanoseconds*1e-9
                    intrinsics=validate_snapshot(rgb,depth,info,header,now)
                    self._snapshot=(rgb,depth,info,header,intrinsics)
                    return True
            except ValueError:
                pass
            time.sleep(0.02)
        return False

    def _depth_to_numpy_meters(self, depth_msg):
        return depth_meters(depth_msg.data, depth_msg.height, depth_msg.width, depth_msg.step, depth_msg.encoding, depth_msg.is_bigendian)

    def _get_K(self):
        return self._snapshot[4]

    def _lookup_T_world_cam(self) -> tuple:
        for world in self.world_frame_candidates:
            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    world,
                    self._snapshot[3].frame_id,
                    rclpy.time.Time.from_msg(self._snapshot[3].stamp),
                    timeout=rclpy.duration.Duration(seconds=1.0),
                )
                t = tf_msg.transform.translation
                q = tf_msg.transform.rotation
                R = _quat_to_rot(float(q.x), float(q.y), float(q.z), float(q.w))
                T = _make_T(R, np.array([t.x, t.y, t.z], dtype=np.float64))
                return T, world
            except Exception:
                continue
        return None, None

    def _robust_depth_at_uv(self, depth_m_img: np.ndarray, u: int, v: int) -> float:
        h, w = depth_m_img.shape[:2]
        r = 10
        u0, u1 = max(0, u - r), min(w - 1, u + r)
        v0, v1 = max(0, v - r), min(h - 1, v + r)

        patch = depth_m_img[v0 : v1 + 1, u0 : u1 + 1]
        selected = self._selected_mask[v0 : v1 + 1, u0 : u1 + 1] > 0
        patch = patch[selected]
        patch = patch[np.isfinite(patch)]
        patch = patch[(patch > 0.05) & (patch < 2.0)]
        if patch.size < 3:
            return None
        return float(np.median(patch))

    def _uv_to_cam_xyz(self, u: int, v: int, depth_m: float) -> np.ndarray:
        fx, fy, cx, cy = self._get_K()
        x = (float(u) - cx) / fx * depth_m
        y = (float(v) - cy) / fy * depth_m
        z = depth_m
        return np.array([x, y, z], dtype=np.float64)

    def _workspace_ok(self, p_world: np.ndarray) -> bool:
        """Conservative workspace gating before IK."""
        if not np.all(np.isfinite(p_world)):
            return False
        x, y, z = float(p_world[0]), float(p_world[1]), float(p_world[2])
        r = math.sqrt(x * x + y * y)

        if z < self.MIN_Z or z > self.MAX_Z:
            self._diag_error(f"[Safety] Z out of range: z={z:.3f}")
            return False

        if r < self.MIN_R or r > self.MAX_R:
            self._diag_error(f"[Safety] Radius out of range: r={r:.3f}")
            return False

        return True

    def _make_topdown_R(self, target_yaw: float) -> np.ndarray:
        """Top-down grasp orientation with TCP z pointing down."""
        cos_yaw = float(np.cos(target_yaw))
        sin_yaw = float(np.sin(target_yaw))

        tcp_x = np.array([cos_yaw, sin_yaw, 0.0], dtype=np.float64)
        tcp_z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        tcp_y = np.cross(tcp_z, tcp_x)
        n = np.linalg.norm(tcp_y)
        if n < 1e-9:
            tcp_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            tcp_y /= n
        tcp_x = np.cross(tcp_y, tcp_z)
        tcp_x /= max(np.linalg.norm(tcp_x), 1e-9)
        return np.column_stack([tcp_x, tcp_y, tcp_z])

    def _solve_ik_safe(self, q_seed: np.ndarray, T_target: np.ndarray, target_yaw: float) -> np.ndarray:
        """Solve IK and run safety filters. Return q or None."""
        ik = inverse_kinematics(q_seed, T_target)
        if ik is None:
            return None

        q = np.asarray(ik["sol"], dtype=np.float64)
        if not np.all(np.isfinite(q)):
            return None
        final = forward_kinematics(q, "tcp")
        if np.linalg.norm(final[:3, 3] - T_target[:3, 3]) > 0.01:
            return None

        if not self._check_solution_safety(q, target_yaw):
            return None

        return q

    # ============================================================
    # Color detection (kept simple)
    # ============================================================

    def detect_cube_uv(self, target_color: str) -> tuple:
        rgb_img = self._snapshot[0] if self._snapshot is not None else None
        if rgb_img is None:
            return None

        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)

        color_ranges = {
            "red": [((0, 80, 60), (10, 255, 255)), ((170, 80, 60), (180, 255, 255))],
            "blue": [((85, 80, 60), (130, 255, 255))],
            "green": [((35, 80, 60), (85, 255, 255))],
        }
        if target_color not in color_ranges:
            return None

        mask = None
        for lo, hi in color_ranges[target_color]:
            m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            mask = m if mask is None else cv2.bitwise_or(mask, m)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 300.0:
            return None

        M = cv2.moments(c)
        if abs(M["m00"]) < 1e-9:
            return None

        u = int(M["m10"] / M["m00"])
        v = int(M["m01"] / M["m00"])
        self._selected_mask = np.zeros_like(mask)
        cv2.drawContours(self._selected_mask, [c], -1, 255, -1)
        return (u, v)

    # ============================================================
    # SAFETY-FIRST grasp: single approach + home lift
    # ============================================================

    def grasp_color(self, target_color: str) -> bool:
        """
        Safety-first markerless grasp (single approach + home lift):
        - Go home
        - Open gripper
        - Detect cube UV + depth
        - Compute p_world
        - Gate workspace
        - Solve ONE IK target (approach-to-grasp)
        - Move slowly to target
        - Close gripper
        - Immediately run 'home' (used as lift)
        """
        self.get_logger().info(f"[Markerless] grasp_color('{target_color}') (single approach + home lift)")

        def retreat():
            try:
                self.gripper_open(duration=0.8)
                time.sleep(0.3)
            except Exception:
                pass
            try:
                self.controller.run_action("home")
            except Exception:
                pass

        # Start from known safe posture
        try:
            self.controller.run_action("home")
            time.sleep(1.5)
        except Exception as e:
            self.get_logger().error(f"[Safety] Failed to run 'home': {e}")
            return False

        # Open gripper
        try:
            self.gripper_open(duration=1.0)
            time.sleep(0.7)
        except Exception:
            pass

        # Camera ready
        if not self._wait_for_rgb_depth_caminfo(timeout_sec=5.0):
            self.get_logger().error("[Safety] RGB/Depth/CameraInfo not ready.")
            return False

        # Detect cube UV
        uv = None
        for _ in range(10):
            time.sleep(0.05)
            uv = self.detect_cube_uv(target_color)
            if uv is not None:
                break
        if uv is None:
            self.get_logger().error(f"[Markerless] Failed to detect cube '{target_color}'")
            return False

        u, v = int(uv[0]), int(uv[1])

        # Depth
        depth_m_img = self._depth_to_numpy_meters(self._snapshot[1])
        z_depth = self._robust_depth_at_uv(depth_m_img, u, v)
        if z_depth is None:
            self.get_logger().error("[Markerless] Invalid depth at target.")
            return False

        p_cam = self._uv_to_cam_xyz(u, v, z_depth)

        # TF world<-cam
        time.sleep(0.05)
        T_world_cam, world_frame = self._lookup_T_world_cam()
        if T_world_cam is None:
            self.get_logger().error("[Markerless] TF lookup failed.")
            return False

        p_world = (T_world_cam @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=np.float64))[:3]
        self.get_logger().info(f"[Raw] p_world: {p_world}")

        # Apply grasp z-offset (stay slightly above raw depth)
        p_target = p_world.copy()
        p_target[2] += self.GRASP_DZ

        # Workspace safety
        if not self._workspace_ok(p_target):
            self.get_logger().error("[Safety] Target outside safe workspace. Aborting.")
            return False

        # Orientation (top-down) and target yaw
        target_yaw = float(np.arctan2(p_target[1], p_target[0] - 0.0251328065010765))
        R_world_tcp = self._make_topdown_R(target_yaw)

        T_target = np.eye(4, dtype=np.float64)
        T_target[:3, :3] = R_world_tcp
        T_target[:3, 3] = p_target

        # Seed
        q_seed = self.get_joint_positions()
        if q_seed is None or len(q_seed) != 5:
            q_seed = np.array([-target_yaw, 0.0, 0.6, 0.8, 0.0], dtype=np.float64)
        else:
            q_seed = np.asarray(q_seed, dtype=np.float64)

        # IK (single)
        self.get_logger().info("[Markerless] Solving IK (single approach)...")
        q_sol = self._solve_ik_safe(q_seed, T_target, target_yaw)
        if q_sol is None:
            self.get_logger().error("[Safety] IK failed or rejected by safety filters.")
            return False

        # Execute
        try:
            self.get_logger().info("[Exec] Move to grasp pose (slow)")
            self.set_joint_positions(q_sol, duration=self.DUR_APPROACH)
            time.sleep(self.DUR_APPROACH + 0.3)

            self.get_logger().info("[Exec] Close gripper")
            self.gripper_close(duration=1.0)
            time.sleep(1.1)

            self.get_logger().info("[Exec] Lift by going home (as requested)")
            self.controller.run_action("home")
            time.sleep(1.2)

            return True

        except Exception as e:
            self.get_logger().error(f"[Safety] Exception during execution: {e}")
            retreat()
            return False

    def place(self, action_name="place_center"):
        """Return action sequence completion, not sensed object release."""
        try:
            return bool(self.controller.run_action(action_name))
        except Exception as error:
            self.get_logger().error(str(error))
            return False

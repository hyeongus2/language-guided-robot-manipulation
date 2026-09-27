import time
import cv2
import rclpy
import numpy as np

from .kinematics import *
from .utils.kinematics_utils import *
from .utils.grasping_base import GraspingNodeBase
from .marker_detector import MarkerDetectionResult


class GraspingNode(GraspingNodeBase):
    def __init__(self, name):
        super().__init__(name)
        self.running = True
        
        self.last_q = None
        self.grasping = False
        self.grasp_targets = []

    ### Utility functions for controlling the robot arm and gripper
    ### DO NOT MODIFY THESE FUNCTIONS

    def gripper_close(self, duration=1.5):
        '''
        Close the gripper.
        '''
        self._set_position_pulse([(10, 550)], duration)

    def gripper_open(self, duration=1.5):
        '''
        Open the gripper.
        '''
        self._set_position_pulse([(10, 100)], duration)

    def get_joint_positions(self):
        '''
        Returns the current joint positions in radians as a numpy array.
        '''
        q = self.get_joint_positions_pulse()
        return pulse2angle(q)
    
    def set_joint_positions(self, q, duration):
        '''
        q: list or numpy array of joint angles (radians)
        duration: time (seconds) to move
        '''
        pulse = angle2pulse(q)
        self.set_joint_positions_pulse(pulse, duration)

    def get_detected_markers(self):
        '''
        Capture and detect ArUco markers from the camera.
        '''
        rclpy.spin_once(self, timeout_sec=0.01)
        if self.image is not None:
            detected_markers = self.marker_detector.detect_markers_with_pose(self.image)
            self.image = None
        else:
            detected_markers = {}
        return detected_markers

    def get_block_pose(
        self,
        detection_result: MarkerDetectionResult,
    ):
        """
        Compute the block's 4x4 homogeneous transformation matrix in the camera frame,
        given the detected marker's rotation (rvec) and translation (tvec) vectors.
        """
        T_cam_marker = np.eye(4, dtype=np.float64)

        # Convert rotation vector (rvec) to a 3x3 rotation matrix
        R, _ = cv2.Rodrigues(detection_result.rvec)
        R = np.asarray(R, dtype=np.float64)

        # Assign rotation and translation
        T_cam_marker[:3, :3] = R
        tvec = np.asarray(detection_result.tvec, dtype=np.float64).reshape(3,)
        T_cam_marker[:3, 3] = tvec

        return T_cam_marker

    def grasp(self, target_marker_id: int | str) -> bool:
        """
        Execute the full grasping procedure for the object with the given marker ID.
        If a string ID is provided, it is mapped through the predefined marker dictionary.
        Returns True if successful, False otherwise.
        """
        is_success = False

        # Use the predefined marker dictionary from grasping_base.py
        if isinstance(target_marker_id, str):
            if target_marker_id in self.marker_ids:
                target_marker_id = self.marker_ids[target_marker_id]
            else:
                self.get_logger().error(f"Unknown marker string: {target_marker_id}")
                return False
            
        self.get_logger().info("Going to home before detection...")
        self.controller.run_action("home")

        # Step 0: Open the gripper
        self.get_logger().info("Opening gripper...")
        self.gripper_open(duration=1.0)
        time.sleep(1.5)

        # Step 1: Detect marker and compute T_cam_marker
        self.get_logger().info(f"Looking for marker {target_marker_id}...")
        detection_result = None
        for _ in range(10):
            detected_markers = self.get_detected_markers()
            if target_marker_id in detected_markers:
                detection_result = detected_markers[target_marker_id]
                self.get_logger().info(f"Marker {target_marker_id} found!")
                break
            time.sleep(0.1)

        if detection_result is None:
            self.get_logger().error(f"Failed to find marker {target_marker_id}.")
            return False
        
        # cam → marker (camera frame to marker frame)
        T_cam_marker = np.asarray(self.get_block_pose(detection_result), dtype=np.float64)

        # Step 2: Compute T_world_cam using current joint angles
        q_current = self.get_joint_positions()
        T_world_cam = np.asarray(forward_kinematics(q_current, 'cam'), dtype=np.float64).copy()
        
        # Step 3: Compute world → marker
        T_world_marker = np.asarray(T_world_cam @ T_cam_marker, dtype=np.float64).copy()

        # Step 4: Define grasp pose - TOP-DOWN APPROACH with gripper aligned to marker axes
        # Extract marker axes in world frame
        R_world_marker = T_world_marker[:3, :3].copy()
        mx = R_world_marker[:, 0].copy()  # Marker X-axis in world
        my = R_world_marker[:, 1].copy()  # Marker Y-axis in world
        mz = R_world_marker[:, 2].copy()  # Marker Z-axis in world (normal to marker surface)

        # TCP Z-axis points downward (opposite of marker Z) - this is the approach direction
        tcp_z = -mz.copy()
        tcp_z = tcp_z / (np.linalg.norm(tcp_z) + 1e-12)

        # TCP X-axis should align with one of the marker axes (X or Y)
        # This determines gripper orientation - gripper fingers will be perpendicular to tcp_x
        # So gripper will grasp along the direction perpendicular to tcp_x
        
        # Choose between marker X and Y axis - pick the one more horizontal (less aligned with world Z)
        world_z = np.array([0., 0., 1.], dtype=np.float64)
        dot_mx = abs(np.dot(mx, world_z))
        dot_my = abs(np.dot(my, world_z))
        
        if dot_mx < dot_my:
            # mx is more horizontal, use it as TCP X
            tcp_x_candidate = mx.copy()
            self.get_logger().info("Gripper aligned with marker X-axis")
        else:
            # my is more horizontal, use it as TCP X  
            tcp_x_candidate = my.copy()
            self.get_logger().info("Gripper aligned with marker Y-axis")
        
        # Project tcp_x_candidate onto plane perpendicular to tcp_z (for orthogonality)
        tcp_x = tcp_x_candidate - np.dot(tcp_x_candidate, tcp_z) * tcp_z
        norm_x = np.linalg.norm(tcp_x)
        
        if norm_x < 1e-9:
            # Fallback - use the other axis
            if dot_mx < dot_my:
                tcp_x = my.copy()
            else:
                tcp_x = mx.copy()
            tcp_x = tcp_x - np.dot(tcp_x, tcp_z) * tcp_z
            norm_x = np.linalg.norm(tcp_x)
        
        tcp_x = tcp_x / (norm_x + 1e-12)
        
        # TCP Y-axis completes the right-handed frame
        tcp_y = np.cross(tcp_z, tcp_x)
        tcp_y = tcp_y / (np.linalg.norm(tcp_y) + 1e-12)
        
        # Re-orthogonalize tcp_x for numerical stability
        tcp_x = np.cross(tcp_y, tcp_z)
        tcp_x = tcp_x / (np.linalg.norm(tcp_x) + 1e-12)

        # Build rotation matrix
        R_world_tcp = np.column_stack([tcp_x, tcp_y, tcp_z])

        # Ensure proper rotation matrix with SVD
        U, _, Vt = np.linalg.svd(R_world_tcp)
        R_world_tcp = U @ Vt
        if np.linalg.det(R_world_tcp) < 0:
            U[:, -1] *= -1.0
            R_world_tcp = U @ Vt

        # Compute target TCP position - closer to the marker
        p_marker = T_world_marker[:3, 3].copy()
        
        # Offset along marker axes for centering
        X_OFFSET = 0.000   # Along marker X-axis
        Y_OFFSET = 0.000   # Along marker Y-axis  
        Z_OFFSET = -0.03   # Slightly above marker along marker Z (reduced from 0.035)
        
        p_world_tcp = (
            p_marker
            + X_OFFSET * mx
            + Y_OFFSET * my
            + Z_OFFSET * mz
        )

        # Build target TCP pose
        T_world_tcp_target = np.eye(4, dtype=np.float64)
        T_world_tcp_target[:3, :3] = R_world_tcp
        T_world_tcp_target[:3, 3] = p_world_tcp

        # Step 5: Solve IK and move to grasp pose
        Q_INIT = np.array([0., 0.4, 1.2, 1.2, 0.], dtype=np.float64)
        
        self.get_logger().info("Solving IK for grasp pose...")
        ik_result = inverse_kinematics(Q_INIT, T_world_tcp_target)
        
        if ik_result is None:
            self.get_logger().error("Failed to solve IK for grasp pose.")
            return False
            
        q_grasp = np.asarray(ik_result["sol"], dtype=np.float64)
        pos_error = ik_result["pos_error"]
        
        self.get_logger().info(f"IK solution found with position error: {pos_error*1000:.2f} mm")
        
        if pos_error > 0.01:  # 10mm threshold
            self.get_logger().warn(f"Large position error: {pos_error*1000:.2f} mm")
        
        self.get_logger().info("Moving to grasp pose...")
        self.set_joint_positions(q_grasp, duration=4.0)
        time.sleep(4.2)

        # Verify achieved pose
        T_achieved = np.asarray(forward_kinematics(q_grasp, 'tcp'), dtype=np.float64).copy()
        actual_error = np.linalg.norm(T_achieved[:3, 3] - T_world_tcp_target[:3, 3])
        self.get_logger().info(f"Actual position error: {actual_error*1000:.2f} mm")

        # Step 6: Close the gripper
        self.get_logger().info("Closing gripper...")
        self.gripper_close(duration=1.0)
        time.sleep(1.2)

        # Step 7: Return to home position
        self.get_logger().info("Moving to home pose...")
        self.controller.run_action("home")

        is_success = True
        return is_success

    def place(self, action_name) -> bool:
        """
        Execute the placing action given an action name.
        Uses predefined action groups if available.
        """
        is_success = False
        
        self.get_logger().info(f"Running place action: {action_name}")
        
        try:
            self.controller.run_action(action_name)
            
            self.get_logger().info("Waiting for place action to complete...")
            time.sleep(2.0)

            self.get_logger().info(f"Place action {action_name} complete.")
            is_success = True
            
        except Exception as e:
            self.get_logger().error(f"Failed to execute place action {action_name}: {e}")
            is_success = False
        
        finally:
            self.controller.run_action("home")

        return is_success

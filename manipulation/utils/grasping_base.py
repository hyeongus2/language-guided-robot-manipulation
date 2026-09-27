"""ROS node adapter. A background executor must service this node."""
import os
import threading
import time
import numpy as np
from rclpy.node import Node
from ros_robot_controller_msgs.srv import GetBusServoState
from ros_robot_controller_msgs.msg import ServoPosition, ServosPosition, GetBusServoCmd
from servo_controller_msgs.msg import ServosPosition as JointPositions
from .action_group_controller import ActionGroupController

class GraspingNodeBase(Node):
    def __init__(self,name):
        super().__init__(name)
        self.last_q=None
        self.servo_client=self.create_client(GetBusServoState,"/ros_robot_controller/bus_servo/get_state")
        if not self.servo_client.wait_for_service(timeout_sec=5.0):raise RuntimeError("Servo state service unavailable")
        self.servo_position_pub=self.create_publisher(ServosPosition,"ros_robot_controller/bus_servo/set_position",1)
        self.joints_pub=self.create_publisher(JointPositions,"servo_controller",1)
        self.controller=ActionGroupController(self.joints_pub,os.environ.get("ROBOT_ACTION_DIR","/home/ubuntu/software/arm_pc/ActionGroups"))

    def get_joint_positions_pulse(self):
        req=GetBusServoState.Request()
        for i in range(1,6):
            cmd=GetBusServoCmd();cmd.id=i;cmd.get_position=1;req.cmd.append(cmd)
        future=self.servo_client.call_async(req)
        completed=threading.Event();future.add_done_callback(lambda _:completed.set())
        if not completed.wait(5):raise TimeoutError("Servo state query timed out")
        response=future.result()
        if response is None:raise RuntimeError("Servo state response missing")
        values=np.array([state.position[0] for state in response.state],dtype=float)
        if values.shape!=(5,) or not np.all(np.isfinite(values)):raise ValueError("Invalid servo state")
        self.last_q=values.copy()
        return values

    def set_joint_positions_pulse(self,pulse,duration):
        self._set_position_pulse(list(enumerate(pulse,1)),duration)

    def _set_position_pulse(self,pulses,duration):
        if not 0.02<=duration<=30:raise ValueError("Motion duration out of range")
        msg=ServosPosition();msg.duration=float(duration)
        for ident,value in pulses:
            if not np.isfinite(value) or not 0<=value<=1000:raise ValueError("Servo pulse outside physical range")
            servo=ServoPosition();servo.id=ident;servo.position=int(value);msg.position.append(servo)
        self.servo_position_pub.publish(msg)

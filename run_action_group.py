import os
from rclpy.node import Node
from servo_controller_msgs.msg import ServosPosition
from manipulation.utils.action_group_controller import ActionGroupController

class ActionGroupExecution(Node):
    def __init__(self,name):
        super().__init__(name)
        pub=self.create_publisher(ServosPosition,"servo_controller",1)
        self.controller=ActionGroupController(pub,os.environ.get("ROBOT_ACTION_DIR","/home/ubuntu/software/arm_pc/ActionGroups"))
    def execute_action_group(self,name):return self.controller.run_action(name)

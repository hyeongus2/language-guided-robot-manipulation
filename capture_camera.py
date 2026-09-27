"""Continuously save the RGB frame consumed by the planning runner."""
import os
from pathlib import Path
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class Capture(Node):
    def __init__(self):
        super().__init__("planning_camera")
        self.bridge=CvBridge()
        self.target=Path(os.environ.get("ROBOT_CAPTURE_DIR",str(Path(__file__).parent/"runtime")))/"capture"/"capture.jpg"
        self.target.parent.mkdir(parents=True,exist_ok=True)
        self.create_subscription(Image,os.environ.get("ROBOT_RGB_TOPIC","/depth_cam/rgb/image_rect_color"),self.receive,1)
    def receive(self,msg):
        frame=self.bridge.imgmsg_to_cv2(msg,desired_encoding="bgr8")
        temp=self.target.with_name("capture.tmp.jpg")
        if not cv2.imwrite(str(temp),frame):raise RuntimeError("Camera file write failed")
        temp.replace(self.target)

def main():
    rclpy.init();node=Capture()
    try:rclpy.spin(node)
    finally:node.destroy_node();rclpy.shutdown()

if __name__=="__main__":main()

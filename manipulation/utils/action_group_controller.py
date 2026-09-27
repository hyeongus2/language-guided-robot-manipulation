"""Adapter for locally installed action-group files (not redistributed)."""
import sqlite3
import time
from pathlib import Path
from servo_controller_msgs.msg import ServosPosition, ServoPosition

class ActionGroupController:
    def __init__(self,pub,action_path):
        self.pub=pub;self.action_path=Path(action_path);self.running_action=False;self.stop_running=False

    def stop_action_group(self):self.stop_running=True

    def run_action(self,name):
        if not name or Path(name).name!=name:raise ValueError("Invalid action name")
        path=self.action_path/(name+".d6a")
        if not path.is_file():raise FileNotFoundError(path)
        if self.running_action:raise RuntimeError("An action is already running")
        self.running_action=True;self.stop_running=False
        try:
            with sqlite3.connect(f"file:{path.as_posix()}?mode=ro",uri=True) as connection:
                rows=0
                for row in connection.execute("SELECT * FROM ActionGroup"):
                    rows+=1
                    if self.stop_running:raise RuntimeError("Action was stopped")
                    duration=float(row[1])/1000
                    if not 0<duration<=30:raise ValueError("Invalid action duration")
                    msg=ServosPosition();msg.position_unit="pulse";msg.duration=duration
                    msg.position=[]
                    for i,pulse in enumerate(row[2:],1):
                        if pulse is None or not 0<=float(pulse)<=1000:raise ValueError("Invalid servo pulse")
                        servo=ServoPosition();servo.id=10 if i==6 else i;servo.position=float(pulse)
                        msg.position.append(servo)
                    self.pub.publish(msg);time.sleep(duration)
                if rows == 0:raise ValueError("Empty action sequence")
            return True
        finally:self.running_action=False

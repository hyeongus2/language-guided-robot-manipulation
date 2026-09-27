#!/usr/bin/env python3
"""Language-guided robot orchestration reconstructed from recorded project files."""

import os
import sys
import re
import ast
import math
import time

import socket
import threading


import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from action_msgs.msg import GoalStatus

from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, PoseStamped
from nav2_msgs.action import NavigateToPose
from tf2_ros import TransformListener, Buffer

from google import genai

# ----------------------------------------------------------------------
# Paths for existing lab code
# ----------------------------------------------------------------------
MANIPULATION_EXPERIMENT_PATH = os.path.dirname(os.path.abspath(__file__))
LLM_PLANNING_PATH = os.environ.get("ROBOT_CAPTURE_DIR", os.path.join(MANIPULATION_EXPERIMENT_PATH, "runtime"))

TASK_DESCRIPTION = "Switch the blue cube and red cube."
TASK_SERVER_HOST = os.environ.get("ROBOT_TASK_HOST", "127.0.0.1")
TASK_SERVER_PORT = 5000

ZONE_IMAGE_CANDIDATES = ["dog", "bike", "cow"]

START_POSE = (-0.06431, 0.04084, -0.03325)

ZONE1_POSE = (0.00000, 1.40000, -3.12967)  # More Down (-X), More Left (+Y)
ZONE2_POSE = (0.00000, -1.05021, 3.02759)  # More Down (-X)
ZONE3_POSE = (1.00611, -0.96932, -0.03317) # Good


if MANIPULATION_EXPERIMENT_PATH not in sys.path:
    sys.path.append(MANIPULATION_EXPERIMENT_PATH)
if LLM_PLANNING_PATH not in sys.path:
    sys.path.append(LLM_PLANNING_PATH)

from run_action_group import ActionGroupExecution  # noqa: E402
from manipulation.grasping import GraspingNode      # noqa: E402

# ----------------------------------------------------------------------
# Gemini (Google GenAI) setup
# ----------------------------------------------------------------------
API_KEY = os.environ.get("API_KEY", None)
if API_KEY is None:
    print("[ERROR] API_KEY environment variable is not set. Please set it before running.")

client = None
if API_KEY is not None:
    client = genai.Client(api_key=API_KEY)


def get_action_from_gemini(image_path: str, prompt: str) -> str:
    """Call Gemini with the given image and prompt; return text response."""
    if client is None:
        return "[ERROR] Gemini client is not initialized. Check API_KEY."

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        return f"[ERROR] Failed to open image: {e}"

    try:
        result = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=[
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": image_bytes}},
            ],
        )

        if hasattr(result, "text") and result.text:
            return result.text

        if not result.candidates:
            return "[ERROR] No candidates from Gemini."

        parts = result.candidates[0].content.parts
        text_segments = []
        for p in parts:
            if hasattr(p, "text") and p.text:
                text_segments.append(p.text)

        if not text_segments:
            return "[ERROR] Empty text response from Gemini."

        return "\n".join(text_segments)

    except Exception as e:
        return f"[ERROR] Exception during Gemini call: {e}"


# ----------------------------------------------------------------------
# Utility functions for pose / quaternion
# ----------------------------------------------------------------------

def yaw_to_quaternion(yaw: float):
    """Convert yaw (rad) to quaternion (x,y,z,w) assuming roll=pitch=0."""
    half_yaw = yaw * 0.5
    qx = 0.0
    qy = 0.0
    qz = math.sin(half_yaw)
    qw = math.cos(half_yaw)
    return qx, qy, qz, qw


def quaternion_to_yaw(q) -> float:
    """Convert quaternion to yaw (rad) assuming roll/pitch ~ 0."""
    x = q.x
    y = q.y
    z = q.z
    w = q.w

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


# ----------------------------------------------------------------------
# Safe parsing helpers
# ----------------------------------------------------------------------

def safe_parse_python_literal(text: str):
    """Safely parse a Python literal using ast.literal_eval."""
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def extract_first_python_list(text: str):
    """Extract the first Python list from a text using regex + ast.literal_eval."""
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return None
    return safe_parse_python_literal(match.group(0))


# ----------------------------------------------------------------------
# Main orchestrator node
# ----------------------------------------------------------------------


class FinalProjectRunner(Node):
    """High-level orchestrator for the final project (v5, Map2 fixed)."""

    def __init__(self, action_node: ActionGroupExecution, grasp_node: GraspingNode):
        super().__init__("final_project_runner")

        self.action_node = action_node
        self.grasp_node = grasp_node

        # Action group names (existing)
        self.ACTION_HOME = "home"
        self.ACTION_LOOK_UP = "look_up"

        # Team 11: Map2 zone images
        self.ZONE_IMAGES = ZONE_IMAGE_CANDIDATES

        # TF2 listener for start pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.amcl_pose_topic = "/amcl_pose"

        # capture_camera.py should continuously update this file.
        self.image_path = os.path.join(LLM_PLANNING_PATH, "capture", "capture.jpg")

        self.task_description = TASK_DESCRIPTION

        # --------------------------------------------------
        # Task Description TCP server (from local Whisper client)
        # - local whisper_client.py sends plain UTF-8 text to (HOST,PORT)
        # - we update self.task_description and unblock the runner
        # --------------------------------------------------
        self._task_lock = threading.Lock()
        self._task_received = threading.Event()
        self._task_server_running = True
        self._task_server_thread = None
        self._start_task_server()


        # Pre-measured Zone1/Zone2/Zone3 poses in the map frame: (x, y, yaw)
        self.ZONE1_POSE = ZONE1_POSE
        self.ZONE2_POSE = ZONE2_POSE
        self.ZONE3_POSE = ZONE3_POSE

        self.held_cube_color = None

        # Exploration results
        self.zone_infos: list[dict] = []
        self.cube_state_by_zone_id: dict[int, str | None] = {}
        self.zone_image_by_zone_id: dict[int, str] = {}
        self.zone_id_by_zone_image: dict[str, int] = {}

        self.start_pose: Pose | None = None
        self.current_pose = None
        self.current_zone_id: int | None = None

        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        self.create_subscription(
            PoseWithCovarianceStamped,
            self.amcl_pose_topic,
            self._amcl_pose_callback,
            10,
        )

        self.plan_steps: list[dict] = []
        self.get_logger().info("FinalProjectRunner(markerless-pick-ready) initialized.")

    # -----------------------------
    # Pose callback
    # -----------------------------

    # -----------------------------
    # Task server (receive task_description from local)
    # -----------------------------

    def _start_task_server(self):
        """Start TCP server thread that updates self.task_description."""
        th = threading.Thread(target=self._task_server_loop, daemon=True)
        self._task_server_thread = th
        th.start()
        self.get_logger().info(
            f"[TaskServer] Listening on {TASK_SERVER_HOST}:{TASK_SERVER_PORT} (send task from local whisper_client.py)"
        )

    def _task_server_loop(self):
        """Accept loop. Ignores empty/whitespace messages."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((TASK_SERVER_HOST, TASK_SERVER_PORT))
            s.listen()
            s.settimeout(1.0)

            while self._task_server_running:
                try:
                    conn, addr = s.accept()
                except socket.timeout:
                    continue
                except Exception as e:
                    self.get_logger().warn(f"[TaskServer] accept() error: {e}")
                    continue

                with conn:
                    self.get_logger().info(f"[TaskServer] Connected: {addr}")
                    while self._task_server_running:
                        try:
                            data = conn.recv(4096)
                        except Exception as e:
                            self.get_logger().warn(f"[TaskServer] recv() error: {e}")
                            break

                        if not data:
                            break

                        text = data.decode("utf-8", errors="ignore").strip()

                        # Must be a real task, not an empty ping/newline
                        if len(text) < 3:
                            try:
                                conn.sendall(b"EMPTY")
                            except Exception:
                                pass
                            continue

                        with self._task_lock:
                            self.task_description = text

                        self.get_logger().info("\n" + "="*60)
                        self.get_logger().info(f"[TaskServer] RECEIVED TASK TEXT:\n{text}")
                        self.get_logger().info("="*60 + "\n")

                        self._task_received.set()
                        self.get_logger().info(f"[TaskServer] Task description updated: {text}")

                        # Keep protocol compatible with your whisper_client.py / robot_server.py
                        try:
                            conn.sendall(b"OK")
                        except Exception:
                            break

    def wait_for_task_description(self):
        """Block forever until task_description arrives (highest priority)."""
        self.get_logger().info(
            "[TaskServer] Waiting for task description... (robot will NOT move until received)"
        )
        self._task_received.clear()
        self._task_received.wait()
        with self._task_lock:
            task = self.task_description
        self.get_logger().info(f"[TaskServer] Using task: {task}")
        return True

    def destroy_node(self):
        """Stop task server thread before destroying ROS node."""
        self._task_server_running = False
        return super().destroy_node()

    def _amcl_pose_callback(self, msg: PoseWithCovarianceStamped):
        self.current_pose = msg.pose.pose

    # -----------------------------
    # Navigation
    # -----------------------------

    def set_start_pose_from_tf(self) -> bool:
        for _ in range(10):
            try:
                trans = self.tf_buffer.lookup_transform(
                    "map",
                    "base_link",
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5),
                )
                x = trans.transform.translation.x
                y = trans.transform.translation.y
                q = trans.transform.rotation
                yaw = quaternion_to_yaw(q)

                pose = Pose()
                pose.position.x = x
                pose.position.y = y
                pose.orientation = q

                self.start_pose = pose
                self.get_logger().info(
                    f"[Phase1] Start pose set from TF: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}"
                )
                return True
            except Exception:
                time.sleep(0.2)

        # Fallback to predefined START_POSE
        pose = Pose()
        x, y, yaw = START_POSE
        pose.position.x = x
        pose.position.y = y
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        self.start_pose = pose
        self.get_logger().error("[Phase1] Could not get TF transform after retries.")
        return False

    def wait_for_nav_server(self, timeout_sec: float = 10.0) -> bool:
        self.get_logger().info("Waiting for NavigateToPose action server ...")
        if not self.nav_client.wait_for_server(timeout_sec=timeout_sec):
            self.get_logger().error("NavigateToPose action server not available!")
            return False
        self.get_logger().info("NavigateToPose action server is ready.")
        return True

    def navigate_to(self, x: float, y: float, yaw: float) -> bool:
        if not self.wait_for_nav_server():
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)

        qx, qy, qz, qw = yaw_to_quaternion(float(yaw))
        goal_msg.pose.pose.orientation.x = qx
        goal_msg.pose.pose.orientation.y = qy
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self.get_logger().info(f"Sending nav goal: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}")
        send_future = self.nav_client.send_goal_async(goal_msg)
        wait_future(send_future, 30.0)
        goal_handle = send_future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("Navigation goal was rejected.")
            return False

        self.get_logger().info("Navigation goal accepted, waiting for result ...")
        result_future = goal_handle.get_result_async()
        try:
            wait_future(result_future, 180.0)
        except TimeoutError:
            goal_handle.cancel_goal_async()
            raise
        result = result_future.result()
        if result is None or result.status != GoalStatus.STATUS_SUCCEEDED:
            return False
        self.get_logger().info(f"Navigation finished with status code: {result.status}")

        if self.current_pose is not None:
            dx = float(self.current_pose.position.x) - float(x)
            dy = float(self.current_pose.position.y) - float(y)
            dist_error = math.hypot(dx, dy)

            current_yaw = quaternion_to_yaw(self.current_pose.orientation)
            raw_yaw_error = current_yaw - float(yaw)
            yaw_error = (raw_yaw_error + math.pi) % (2.0 * math.pi) - math.pi

            self.get_logger().info(
                f"Final pose error: dist={dist_error:.3f} m, yaw_error={yaw_error:.3f} rad"
            )

        return True

    def run_action_group_safe(self, name):
        if self.action_node is None:
            return False
        try:
            return bool(self.action_node.execute_action_group(name))
        except Exception as error:
            self.get_logger().error(str(error))
            return False

    # -----------------------------
    # Zone pose mapping (Zone1/2/3 are fixed)
    # -----------------------------

    def get_zone_pose_from_id(self, zone_id: int):
        if zone_id == 1:
            return self.ZONE1_POSE
        if zone_id == 2:
            return self.ZONE2_POSE
        if zone_id == 3:
            return self.ZONE3_POSE
        raise ValueError(f"Unknown zone_id: {zone_id}")

    # -----------------------------
    # Phase 2: Exploration (zone image + cube color)
    # -----------------------------

    def _build_zone_image_prompt(self, zone_id: int) -> str:
        candidates = ", ".join([f"\"{z}\"" for z in self.ZONE_IMAGES])
        return f"""
You are a vision assistant for a mobile robot.

The robot is looking at the WALL of a single zone (Zone {zone_id}).
On the wall, there is exactly ONE large picture that identifies the zone.

The zone image is exactly one of the following candidates: {candidates}.

Your task:
  1) Ignore anything on the floor.
  2) Identify which candidate image appears on the wall.
  3) Output the zone_image string.

Return ONLY a Python dictionary in exactly this format:
{{"zone_image": "dog"}}

Do NOT add any explanation or extra text.
"""

    def _build_cube_color_prompt(self, zone_id: int) -> str:
        return f"""
You are a vision assistant for a mobile robot.

The robot is looking at the FLOOR area in front of Zone {zone_id}.
There can be at most ONE colored cube placed on the floor in front of the zone.

Your task:
  1) Ignore any wall images.
  2) Look only at the cube on the floor.
  3) The cube color is exactly one of: "blue", "green", "red".
  4) If there is NO cube on the floor, use the string "none".

Return ONLY a Python dictionary in exactly this format:
{{"cube_color": "blue"}}
or
{{"cube_color": "none"}}

Do NOT add any explanation or extra text.
"""

    def _detect_zone_image_with_gemini(self, zone_id: int) -> str | None:
        prompt = self._build_zone_image_prompt(zone_id)
        max_retries = 3

        for attempt in range(1, max_retries + 1):
            self.get_logger().info(
                f"[ZoneImage] Calling Gemini for Zone {zone_id}, attempt {attempt}/{max_retries} ..."
            )
            resp = get_action_from_gemini(self.image_path, prompt)
            self.get_logger().info(f"[ZoneImage] Gemini response for Zone {zone_id}: {resp}")

            m = re.search(r"\{[\s\S]*\}", resp)
            if not m:
                self.get_logger().warn("[ZoneImage] No dict found, retrying.")
                time.sleep(1.0)
                continue

            obj = safe_parse_python_literal(m.group(0))
            if not isinstance(obj, dict) or "zone_image" not in obj:
                self.get_logger().warn("[ZoneImage] Parsed object invalid, retrying.")
                time.sleep(1.0)
                continue

            zone_image = str(obj["zone_image"]).strip().lower()
            if zone_image not in self.ZONE_IMAGES:
                self.get_logger().warn(
                    f"[ZoneImage] zone_image='{zone_image}' not in allowed set {self.ZONE_IMAGES}. Retrying."
                )
                time.sleep(1.0)
                continue

            return zone_image

        self.get_logger().error(
            f"[ZoneImage] Failed to detect zone_image for Zone {zone_id} after {max_retries} attempts."
        )
        return None

    def _detect_cube_color_with_gemini(self, zone_id: int) -> str | None:
        prompt = self._build_cube_color_prompt(zone_id)
        max_retries = 3

        for attempt in range(1, max_retries + 1):
            self.get_logger().info(
                f"[CubeColor] Calling Gemini for Zone {zone_id}, attempt {attempt}/{max_retries} ..."
            )
            resp = get_action_from_gemini(self.image_path, prompt)
            self.get_logger().info(f"[CubeColor] Gemini response for Zone {zone_id}: {resp}")

            m = re.search(r"\{[\s\S]*\}", resp)
            if not m:
                self.get_logger().warn("[CubeColor] No dict found, retrying.")
                time.sleep(1.0)
                continue

            obj = safe_parse_python_literal(m.group(0))
            if not isinstance(obj, dict) or "cube_color" not in obj:
                self.get_logger().warn("[CubeColor] Parsed object invalid, retrying.")
                time.sleep(1.0)
                continue

            color = str(obj["cube_color"]).strip().lower()
            if color == "none":
                return None
            if color not in ("red", "green", "blue"):
                self.get_logger().warn(f"[CubeColor] Unexpected color='{color}', retrying.")
                time.sleep(1.0)
                continue

            return color

        self.get_logger().warn(f"[CubeColor] Failed to detect cube color for Zone {zone_id}. Treat as none.")
        return None

    def capture_and_analyze_zone(self, zone_id: int):
        self.get_logger().info(f"--- Exploring Zone {zone_id} ---")
        x, y, yaw = self.get_zone_pose_from_id(zone_id)
        nav_success = self.navigate_to(x, y, yaw)
        if not nav_success:
            self.get_logger().warn(f"Navigation to Zone {zone_id} failed. Skipping.")
            return None

        self.get_logger().info(f"[Zone {zone_id}] Looking UP to detect zone image.")
        if not self.run_action_group_safe(self.ACTION_LOOK_UP):
            raise RuntimeError("look_up action did not complete")
        time.sleep(2.0)

        zone_image = self._detect_zone_image_with_gemini(zone_id)
        if zone_image is None:
            self.get_logger().error(f"[Zone {zone_id}] Zone image detection failed. Skipping.")
            return None

        self.get_logger().info(f"[Zone {zone_id}] Going to HOME pose to detect cube color.")
        if not self.run_action_group_safe(self.ACTION_HOME):
            raise RuntimeError("home action did not complete")
        time.sleep(2.0)

        cube_color = self._detect_cube_color_with_gemini(zone_id)

        self.get_logger().info(
            f"Zone {zone_id} -> zone_image={zone_image}, cube_color={cube_color}"
        )

        return {
            "zone_id": int(zone_id),
            "zone_image": zone_image,
            "cube_color": cube_color,
        }

    def run_phase2_exploration(self):
        self.get_logger().info("=== Phase 2: Zone image & cube exploration ===")

        self.zone_infos = []
        self.zone_image_by_zone_id = {}
        self.zone_id_by_zone_image = {}
        self.cube_state_by_zone_id = {}

        for zone_id in (1, 2, 3):
            info = self.capture_and_analyze_zone(zone_id)
            if info is None:
                continue

            self.zone_infos.append(info)

            z_id = int(info["zone_id"])
            z_img = str(info["zone_image"]).strip().lower()
            c_col = info["cube_color"]

            self.zone_image_by_zone_id[z_id] = z_img
            if z_img not in self.zone_id_by_zone_image:
                self.zone_id_by_zone_image[z_img] = z_id
            else:
                self.get_logger().warn(
                    f"Duplicate zone_image '{z_img}' detected (already mapped to Zone {self.zone_id_by_zone_image[z_img]}). "
                    f"Keeping the first mapping; Zone {z_id} will not be used for '{z_img}'."
                )

            self.cube_state_by_zone_id[z_id] = c_col

        self.get_logger().info(f"Collected zone_infos: {self.zone_infos}")
        self.get_logger().info(f"zone_image_by_zone_id: {self.zone_image_by_zone_id}")
        self.get_logger().info(f"zone_id_by_zone_image: {self.zone_id_by_zone_image}")
        self.get_logger().info(f"cube_state_by_zone_id: {self.cube_state_by_zone_id}")

    # -----------------------------
    # Phase 3: Planning via LLM
    # -----------------------------

    def build_environment_description(self) -> str:
        lines = []
        for zone_id in (1, 2, 3):
            zone_image = self.zone_image_by_zone_id.get(zone_id, "unknown")
            cube_color = self.cube_state_by_zone_id.get(zone_id, None)
            cube_text = "no cube" if cube_color is None else f"a {cube_color} cube"
            lines.append(
                f"The zone with image '{zone_image}' currently contains {cube_text}."
            )
        return "\n".join(lines)

    def build_task_prompt(self) -> str:
        env_text = self.build_environment_description()

        task_text = f"""
Task description (given by the TA on the demo day):

{self.task_description}

Environment facts:
- There are exactly two cubes in total (among blue, red, and green).
- Initially, each zone contains at most one cube.
- Therefore, at the beginning, exactly one zone is empty.
- During the task, it is allowed that more than one cube ends up in the same zone.
"""

        candidates = ", ".join([f"\"{z}\"" for z in self.ZONE_IMAGES])

        plan_instructions = f"""
You will output a sequence of high-level robot actions in a Python list.

Available actions:
    - {{"action": "go", "zone_image": Z}}
    - {{"action": "pick"}}
    - {{"action": "place"}}

Rules:
    1) "go" moves the robot base in front of the specified zone image Z.
    2) Z must be exactly one of: {candidates}.
    3) "pick" always picks up the cube in the CURRENT zone (if any).
    4) "place" always places the currently held cube in the CURRENT zone.
    5) The robot can hold at most one cube at a time.
    6) The sequence must solve the task exactly, respecting the actual cube colors.

Return ONLY a Python list of dictionaries, for example:
[
  {{"action": "go", "zone_image": "dog"}},
  {{"action": "pick"}},
  {{"action": "go", "zone_image": "bike"}},
  {{"action": "place"}}
]

Do not add any explanation or extra text outside the Python list.
"""

        return f"""
[Environment description]
{env_text}

[Task]
{task_text}

[Planning instructions]
{plan_instructions}
"""

    def run_phase3_planning(self):
        self.get_logger().info("=== Phase 3: Planning via Gemini ===")

        prompt = self.build_task_prompt()
        self.get_logger().info("Sending planning prompt to Gemini ...")
        response = get_action_from_gemini(self.image_path, prompt)
        self.get_logger().info(f"Gemini planning response:\n{response}")

        steps = extract_first_python_list(response)
        if steps is None or not isinstance(steps, list):
            self.get_logger().error("Failed to parse plan from Gemini response.")
            self.plan_steps = []
            return

        for step in steps:
            if not isinstance(step, dict) or step.get("action") not in {"go", "pick", "place"}:
                raise ValueError("Invalid plan action")
            if step["action"] == "go" and str(step.get("zone_image", "")).strip().lower() not in self.zone_id_by_zone_image:
                raise ValueError("Unknown plan destination")
        self.plan_steps = steps
        self.get_logger().info(f"Parsed plan_steps: {self.plan_steps}")

    # -----------------------------
    # Phase 4: Execution
    # -----------------------------

    def pick_cube(self):
        if self.current_zone_id is None or self.held_cube_color is not None:
            return False
        color=self.cube_state_by_zone_id.get(self.current_zone_id)
        if color is None:
            return False
        mode=os.environ.get("ROBOT_GRASP_MODE", "markerless")
        if mode == "fixed":
            success=self.run_action_group_safe("pick_center")
        elif mode == "markerless":
            success=bool(self.grasp_node.grasp_color(color))
        else:
            raise ValueError("ROBOT_GRASP_MODE must be markerless or fixed")
        if success:
            self.held_cube_color=color
            self.cube_state_by_zone_id[self.current_zone_id]=None
        return success

    def place_cube(self) -> bool:
        """Place the currently held cube at the current zone using a predefined action group."""
        if self.current_zone_id is None:
            self.get_logger().error("Current zone id is None. Cannot place.")
            return False

        if self.held_cube_color is None:
            self.get_logger().warn("No cube is currently held. Place is skipped.")
            return False

        zone_id = self.current_zone_id
        color = self.held_cube_color

        action_name = "place_center"
        self.get_logger().info(f"[PLACE] Zone {zone_id}, cube_color={color}, action_name={action_name}")

        success = bool(self.grasp_node.place(action_name))
        self.get_logger().info(f"[PLACE] Result: {success}")

        if success:
            self.cube_state_by_zone_id[zone_id] = color
            self.held_cube_color = None

        return success

    def run_phase4_execution(self):
        self.get_logger().info("=== Phase 4: Plan execution ===")

        for step in self.plan_steps:
            if not isinstance(step, dict):
                self.get_logger().warn(f"Skipping invalid plan step (not a dict): {step}")
                continue

            action = step.get("action")

            if action == "go":
                zone_image = step.get("zone_image")
                if zone_image is None:
                    self.get_logger().warn(f"[GO] Missing zone_image in step: {step}")
                    continue

                z_img = str(zone_image).strip().lower()
                zone_id = self.zone_id_by_zone_image.get(z_img)
                if zone_id is None:
                    self.get_logger().warn(
                        f"[GO] Unknown zone_image='{z_img}'. Known: {list(self.zone_id_by_zone_image.keys())}"
                    )
                    continue

                pose = self.get_zone_pose_from_id(zone_id)
                self.get_logger().info(f"[GO] Moving to zone_image='{z_img}' (Zone {zone_id}) at pose {pose}")
                x, y, yaw = pose
                if not self.navigate_to(x, y, yaw):
                    return False
                self.current_zone_id = zone_id

            elif action == "pick":
                success = self.pick_cube()
                if not success:
                    self.get_logger().error("[PLAN] Pick failed; stopping plan.")
                    return False

            elif action == "place":
                success = self.place_cube()
                if not success:
                    self.get_logger().error("[PLAN] Place failed; stopping plan.")
                    return False

            else:
                self.get_logger().warn(f"Unknown action in plan: {action}")

    # -----------------------------
    # Phase 5: Return to start
    # -----------------------------

    def run_phase5_return_to_start(self):
        if self.start_pose is None:
            self.get_logger().warn("Start pose is None. Skipping return-to-start phase.")
            return

        x = self.start_pose.position.x
        y = self.start_pose.position.y
        yaw = quaternion_to_yaw(self.start_pose.orientation)
        self.get_logger().info(
            f"=== Phase 5: Returning to start pose (x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}) ==="
        )
        self.navigate_to(x, y, yaw)

    # -----------------------------
    # Orchestration
    # -----------------------------

    def run_all_phases(self):
        self.get_logger().info("=== Final Project Runner started ===")

        # Task is the top priority: do not move until we receive it from local
        self.wait_for_task_description()

        # Now it is safe to move / explore / plan
        self.get_logger().info("[Phase0] Going to LOOK UP pose before everything.")
        if not self.run_action_group_safe(self.ACTION_LOOK_UP):
            raise RuntimeError("look_up action did not complete")

        self.set_start_pose_from_tf()

        self.run_phase2_exploration()
        self.run_phase3_planning()
        self.run_phase4_execution()
        self.run_phase5_return_to_start()

        self.get_logger().info("=== Final Project Runner finished ===")
        

def wait_future(future, timeout):
    done=threading.Event()
    future.add_done_callback(lambda _:done.set())
    if not done.wait(timeout):
        raise TimeoutError("ROS operation timed out")
    if future.exception() is not None:
        raise future.exception()

def main():
    rclpy.init()
    nodes=[];executor=MultiThreadedExecutor(num_threads=4);thread=None
    try:
        action_node=ActionGroupExecution("action_group_runner");nodes.append(action_node)
        grasp_node=GraspingNode("grasping_node_runner");nodes.append(grasp_node)
        runner=FinalProjectRunner(action_node,grasp_node);nodes.append(runner)
        for node in nodes:executor.add_node(node)
        thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
        runner.run_all_phases()
    finally:
        for node in nodes:
            if hasattr(node,"_task_server_running"):node._task_server_running=False
        executor.shutdown()
        if thread is not None:thread.join(timeout=5)
        for node in reversed(nodes):node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


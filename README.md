# 언어 지시 기반 로봇 인지·계획·조작

2025년 가을 전자설계실험에서 음성 지시를 장면 관찰, 작업 계획, 이동, 팔 제어로 연결하는 프로젝트를 진행했습니다. 한 학기 동안 인지·경로 계획·조작·강화학습을 실습하고, 팀 프로젝트에서 각 모듈의 실행 흐름을 통합했습니다.

## 구조

```text
Whisper 음성 인식 → TCP 지시 전달 → 구역 탐색·영상 관찰
                                      ↓
                         Gemini 작업 계획 → Nav2 이동
                                      ↓
                             팔 제어·작업 상태 갱신
```

| 모듈 | 역할 |
|---|---|
| `final_project_runner.py` | 구역 인식, 작업 계획, navigation, pick/place 상태 연결 |
| `manipulation/grasping.py` | 색상 mask와 정렬 RGB-depth를 이용한 물체 위치 추정, TF 변환, markerless 조작 |
| `manipulation/kinematics.py` | 팔 형상에 따른 FK, 관절 범위와 위치 오차를 고려한 IK |
| `labs/aruco_grasping.py` | ArUco marker 기반 집기 실습 |

ROS callback은 background executor에서 처리합니다. Navigation 성공 후 상태를 갱신하고, pick/place 실패 시 다음 계획 실행을 중단합니다. 깊이 영상의 형식·프레임·시각·카메라 파라미터를 검사하며, IK 해는 관절 범위와 최종 FK 오차를 확인한 뒤 사용합니다.

## 환경과 실행

Linux, Python 3.10+, ROS 2, Nav2, OpenCV가 필요합니다. ROS 패키지는 `rclpy`, `sensor_msgs`, `geometry_msgs`, `action_msgs`, `tf2_ros`, `cv_bridge`, `ros_robot_controller_msgs`, `servo_controller_msgs`를 사용합니다. 로봇 SDK와 보정된 action 파일은 장비 환경에서 별도로 준비합니다.

로봇의 ROS 환경을 source한 뒤 실행합니다.

```bash
python -m pip install -e .
export ROBOT_ACTION_DIR=/path/to/ActionGroups
export API_KEY=YOUR_LOCAL_API_KEY
export ROBOT_GRASP_MODE=markerless
robot-planning-camera
# 별도 터미널에서도 같은 ROS 환경과 환경변수 사용
robot-task-runner
```

### 카메라와 팔 설정

- Rectified RGB와 RGB에 정렬된 depth, CameraInfo의 프레임·해상도를 맞추고 `ROBOT_ALIGNED_RGB_DEPTH=1`을 설정합니다.
- 기본 topic은 `/depth_cam/rgb/image_rect_color`, `/depth_cam/aligned_depth_to_color/image_raw`, `/depth_cam/rgb/camera_info`입니다. `ROBOT_RGB_TOPIC`, `ROBOT_DEPTH_TOPIC`, `ROBOT_INFO_TOPIC`으로 변경할 수 있습니다.
- IK 기준 프레임은 `base_footprint`입니다. 팔 영점, TCP, 작업 영역, 집기 높이와 구역 좌표를 장비·지도에 맞춰 설정합니다.
- `home.d6a`, `look_up.d6a`, `place_center.d6a`는 장비에서 보정한 파일을 사용합니다. `ROBOT_GRASP_MODE=fixed`는 `pick_center.d6a`의 사전 정의 동작을 실행합니다.
- Pick/place의 `True` 반환은 명령 시퀀스 완료를 나타냅니다. 물체 보유 여부를 판정하는 센서 피드백은 구현하지 않았습니다.

### 음성 클라이언트

별도 PC에서 `requirements-client.txt`를 설치하고 `ROBOT_IP`를 설정한 뒤 `python whisper_client.py`를 실행합니다. Runner의 기본 수신 주소는 loopback입니다. PC 간 연결에는 실습망 인터페이스를 `ROBOT_TASK_HOST`로 지정합니다.

## 테스트

```bash
python -m unittest discover -s tests -v
```

장비 없이 실행하는 검사 **10개를 통과**했습니다. 깊이 영상의 16UC1/32FC1·endianness·stride, 프레임과 시간 동기화, FK/IK 왕복, 도달 불가능한 목표, navigation 실패와 action 파일 누락을 검사합니다.

통합 실행에는 실제 카메라·TF·servo·Nav2·Gemini API 검증이 필요합니다. 자동 테스트는 이 장비·서비스 통합을 포함하지 않습니다.

## 실습 시연

- [2025-11-01 ArUco marker 기반 집기·놓기](https://drive.google.com/file/d/1jFSeSVtTVsvS1iWjP8NCCWp-PvL8udkh/view)
- [2025-12-05 Offline RL 로봇팔 실습](https://drive.google.com/file/d/1RbyEvCyg_KMFjWTKeGe2Cddf7d9G8XIx/view)

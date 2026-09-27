# 언어 지시 기반 로봇 인지·계획·조작

2025년 가을 전자설계실험에서 한 학기 동안 인지·경로 계획·팔 제어·강화학습을 실습하고, 최종 프로젝트에서 음성/언어 지시를 구역 탐색·물체 인식·작업 계획·이동·조작으로 연결했습니다. 팀 종합 시연에서 언어 지시부터 이어지는 실행을 구성했습니다.

## 구성

```text
Whisper 음성 인식 → TCP 지시 전달 → 구역 탐색/영상
                                      ↓
                         Gemini 작업 계획 → Nav2 이동
                                      ↓
                         markerless 또는 fixed 조작
```

`final_project_runner.py`는 구역 인식·LLM 계획·navigation·pick/place 상태를 연결합니다. `manipulation/grasping.py`는 색상 mask, 정렬 RGB-depth, TF, IK를 이용한 markerless 구현입니다. `manipulation/kinematics.py`는 기록된 팔 형상에 대한 FK와 관절 범위를 적용한 IK입니다. 기록 파일에서 공개 실행 패키지로 재구성했으며 당시 로봇 설치본과 파일 배치가 동일하다는 뜻은 아닙니다.

학기 중 ArUco 집기 실습은 `labs/aruco_grasping.py`에 별도 기록으로 보존합니다. 이 파일은 과거 실습 패키지의 marker detector·환경에 의존하는 참고 구현이며 현재 runner 진입점이 아닙니다. 최종 종합 시연에서는 당시 허락을 받고 집기 단계만 fixed action으로 바꿨습니다. markerless 구현의 존재, ArUco lab 영상, fixed 종합 시연을 서로 구분합니다. RL lab도 학기 실습의 일부이며 최종 markerless 성공 영상으로 표시하지 않습니다.

## 환경과 실행

Linux, Python 3.10+, ROS 2, Nav2, `rclpy`, `sensor_msgs`, `geometry_msgs`, `action_msgs`, `tf2_ros`, `cv_bridge`, OpenCV와 로봇에 설치된 `ros_robot_controller_msgs`·`servo_controller_msgs`가 필요합니다. 제공 하드웨어 SDK·URDF mesh·강의 배포물·action 파일은 이 저장소에 복제하지 않습니다. 기존 로봇 환경의 setup을 source하고 Python 패키지를 설치합니다.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
export ROBOT_ACTION_DIR=/path/to/ActionGroups
export API_KEY=YOUR_LOCAL_API_KEY
export ROBOT_GRASP_MODE=markerless
robot-planning-camera
# 별도 터미널에서 ROS 환경과 같은 환경변수를 설정
robot-task-runner
```

`home.d6a`, `look_up.d6a`, `place_center.d6a`와 fixed 모드의 `pick_center.d6a`는 실제 로봇에서 보정한 파일을 사용합니다. `ROBOT_GRASP_MODE=fixed`를 선택하면 기록된 fixed 시연 동작을 별도로 사용할 수 있습니다. 반환 True는 명령 시퀀스 완료이며 실제 물체 보유 센서의 판정이 아닙니다.

markerless 실행 전 rectified RGB와 RGB에 정렬된 depth 및 CameraInfo의 프레임·해상도를 확인하고 `ROBOT_ALIGNED_RGB_DEPTH=1`을 지정합니다. 기본 topic은 `/depth_cam/rgb/image_rect_color`, `/depth_cam/aligned_depth_to_color/image_raw`, `/depth_cam/rgb/camera_info`이며 `ROBOT_RGB_TOPIC`, `ROBOT_DEPTH_TOPIC`, `ROBOT_INFO_TOPIC`으로 바꿉니다. IK 기준은 기록된 모델과 같은 `base_footprint`입니다. 팔 영점·TCP·작업 영역·집기 높이는 실제 장비에 맞춰 확인해야 합니다.

음성 클라이언트는 `requirements-client.txt`를 별도 PC에 설치하고 `ROBOT_IP`를 설정해 `python whisper_client.py`로 실행합니다. runner의 수신 기본 주소는 loopback입니다. 서로 다른 PC에서 사용할 때는 신뢰하는 실습망의 인터페이스를 `ROBOT_TASK_HOST`로 지정합니다. 예제의 구역 좌표는 원 실습 지도 기준이므로 다른 지도에서는 수정해야 합니다.

## 공개판 수정과 검증

ROS 초기화를 진입점 하나로 모으고 background executor가 runner와 grasp callback을 처리하도록 구성했습니다. navigation 성공 확인 뒤 상태를 바꾸고 pick/place 실패 시 계획을 멈춥니다. action 부재·중단·servo 조회 실패를 명시적으로 전달합니다. depth의 16UC1/32FC1·endianness·stride를 구분하고 프레임·시각·intrinsics를 검사합니다. IK 후 관절값을 잘라 실행하지 않고 bounds와 최종 FK 오차를 확인합니다.

현재 확인한 결과는 Python 문법, 깊이 변환·동기화 계약, FK/IK 합성 왕복 등의 장비 없는 검사입니다. ROS graph·실제 카메라·servo·Nav2·Gemini API와 로봇 집기는 이 공개판으로 다시 시험하지 않았습니다. 한 학기 수행과 당시 시연은 과거 경험이며 공개판의 실기기 재검증과 구분합니다.

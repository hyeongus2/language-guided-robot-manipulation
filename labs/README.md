# 로봇 인지·조작 실습

2025년 가을 학기에는 FK/IK, ArUco 기반 조작, 시각 인지, 이동·언어 계획, Offline RL/IQL을 실습했습니다.

## ArUco 조작

`aruco_grasping.py`는 marker의 위치를 이용해 집기 동작을 구성하는 실습 코드입니다. 실행에는 실습 패키지의 marker detector와 상대 import 모듈이 필요합니다. 이 모듈은 최종 runner의 markerless 조작 모듈과 별도로 사용합니다.

## 의존성

IQL framework, R3M weights, YOLO 모델은 각 배포처의 이용 조건에 따라 별도로 준비합니다. 이 폴더에는 해당 framework와 weights를 포함하지 않습니다.

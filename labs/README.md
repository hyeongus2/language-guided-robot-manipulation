# 학기 실습 기록

한 학기 동안 FK/IK와 ArUco 기반 조작, 시각 인지, 이동 계획, 언어 계획, offline RL/IQL 실습을 수행했습니다. 보존된 ArUco 조작 소스는 `aruco_grasping.py`입니다. 이는 당시 실습 패키지의 상대 import·marker detector에 의존하는 참고 소스이며 현재 재구성한 markerless package에 직접 import하는 실행 진입점이 아닙니다.

본 저장소는 최종 통합 runner와 공개판 계산·제어 코드를 중심으로 구성합니다. 수업 제공 IQL framework·R3M weights·YOLO model·교재를 일괄 재배포하지 않습니다. 학기 실습의 완료를 모든 기법의 독자 개발이나 실기기 재현 성공으로 확대하지 않습니다.

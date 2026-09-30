Changelog
=========

2026-09-14
----------

Changed
~~~~~~~
- unitree_rl_lab ``g1_29dof_running`` (2026-09-11_18-14-36) 정책 배포용으로 RL 파트 교체
  - 29-DoF 전신 액션, 관측 98 x 히스토리 5 = 490 (term-major, 과거→현재)
  - 조인트 매핑을 ``joint_ids_map`` (deploy.yaml) 하나로 통일
  - RL 상태에서 상체도 정책 출력 + ``arm_waist_kps/kds`` 적용 (기존: 기본자세 + init 게인 고정)
  - gait clock ``gait_period`` / ``cmd_threshold`` YAML 설정화 (기존 0.8s 하드코딩)
  - 속도 명령을 ``cmd_min`` / ``cmd_max`` 비대칭 범위로 클램프 (``max_cmd`` 대체)
- ONNX Runtime >= 1.14 사용 (``GetInputNameAllocated``), 기본 경로를 unitree_rl_lab prebuilt 1.22.0으로 지정
  - 모델 로드 시 입력/출력 차원 검증, 불일치 시 즉시 종료
- 토크 추정(클램프/로그)을 ``rl_kp`` 고정값 대신 실제 전송 중인 kp/kd로 계산
- 추론 실패 시 DAMPING 전환 (기존: 빈 출력 역참조)
- 토크 클램프와 킬 임계값 분리
  - 킬(``torque_limit``): 모터 피크 스펙 = 학습 ``effort_limit_sim`` = MJCF ctrlrange
    (다리 80/120 -> 88/139). 기존 값은 하드웨어보다 낮아 정상 포화가 곧 종료였음
  - 클램프: 킬의 97%(``TORQUE_CLAMP_RATIO``, 무릎 135 Nm). 간격이 tau_est 노이즈와
    500Hz 추정 지연을 흡수하므로, 킬은 클램프 실패/모터 이상일 때만 발동
- 로그에 진단 컬럼 13개 추가 (기존 146개 컬럼 뒤에 붙임, 기존 분석 스크립트 호환)
  - ``cmd_x/y/w``: 정책에 실제로 들어간 속도 명령 -> 속도별 토크 프로파일 분석 가능
  - ``phase_sin/cos``: gait 클록. (0,0)이면 정지 모드
  - ``roll/pitch/yaw``, ``gyro_x/y/z``: 골반 IMU -> 낙상 원인/시점 분석
  - ``fsm_state``: 0 wait_init / 1 moving / 2 wait_policy / 3 rl_active / 4 damping
  - ``clamp_mask``: 직전 기록 이후 토크 클램프가 걸린 모터 비트마스크
- 상체 관절 속도 킬 임계값은 7 rad/s 유지 (원본 그대로)
  - 단, 정책이 팔을 구동하므로 정상 팔스윙에서 트립 예상 (sim2sim 실측 8.3~9.0 rad/s)

2025-12-26
----------

Changed
~~~~~~~
- 필요없는 내용, 변수들 정리 및 하드코딩 최소화
  - RL_POLICY_WAVE_HAND 관련 코드 제거
  - G1_NUM_MOTOR 상수 분리(G1_NUM_LEG_MOTOR, G1_NUM_UPPER_MOTOR)


2025-12-25
----------

Changed
~~~~~~~
- CMake thirdparty 구조 정리(yaml-cpp subdirectory 빌드, onnxruntime imported target)
- 로그 파일명에 타임스탬프 적용(YYYY-MM-DD_HH-MM-SS)

Fixed
~~~~~
- onnxruntime include 경로 누락으로 인한 컴파일 에러 해결
- 로그 디렉토리 미생성 시 open 실패 → 디렉토리 생성 로직 추가

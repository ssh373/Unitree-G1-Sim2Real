Changelog
=========

2026-08-07
----------

Changed
~~~~~~~
- 제어 / 통신 분리를 정리하고 이름을 확정

  - ``wholebody_rl``: 제어만 — 상태 머신, 관측, ONNX 추론. DDS·MuJoCo 헤더 없음
  - ``UnitreeComm``: 실기 DDS (``rt/lowstate`` ↔ ``rt/lowcmd``), CRC, 게임패드, 500Hz 루프
  - ``MujocoSim``: MuJoCo 물리, ``d->ctrl`` 에 PD 토크, 접지 높이 탐색, 베이스 고정
  - ``MujocoViewer``: GLFW 창·카메라·키보드 (신규)
  - 시임: ``read()`` / ``set_input()`` / ``control()`` / ``write()``, 루프는 호출자 소유
  - 공유 어휘는 ``robot_types.hpp``

- MuJoCo 뷰어를 직접 구현으로 교체

  - MuJoCo ``simulate`` UI 벤더링(업스트림 11,115줄 + lodepng)을 제거
  - 의존성이 MuJoCo + GLFW + OpenGL 로 축소, 스레드 2개 → 1개
  - 렌더링은 여전히 libmujoco 의 ``mjv_*`` / ``mjr_*``

- 시뮬 조작을 키보드로: 방향키(또는 W/S/Q/E) 이동·회전, A/D 게걸음, shift 미세조정,
  X 0, space 일시정지, backspace 에피소드 재시작. 명령은 latch 방식이라 -1..1 임의 값 가능
- onnxruntime 1.12 이상만 지원 (``ORT_API_VERSION`` 분기 제거, 미달 시 ``#error``)

Removed
~~~~~~~
- 미사용 코드 정리: ``Quat2RPY``, ``stride_a`` / ``stride_b`` / ``eps``, ``cout_mutex``,
  ``MujocoSim::set_policy_at`` / ``base_height``, ``G1_NUM_UPPER_MOTOR``,
  ``ImuState::rpy``, ``CommandInput::mode``

Fixed
~~~~~
- simulate UI 시절 남아 있던 문제들이 단일 스레드 재작성으로 해소
  (락 경합으로 화면이 몇 초에 한 번 갱신되던 문제, Reset 후 베이스가 고정되던 문제)

2026-08-04
----------

Changed
~~~~~~~
- 로봇 통신 코드와 공통 제어 코드 분리 (하드웨어 추상화)

  - ``wholebody_rl``: ``read()`` / ``set_input()`` / ``control()`` / ``write()`` 시임만 노출,
    DDS·MuJoCo를 전혀 모름. 루프 소유권은 백엔드로 이동
  - ``unitree_backend``: DDS, CRC, ``mode_machine``, 게임패드, 500/50Hz 스레드,
    모터 드라이버 토크 리밋 트릭
  - ``mujoco_backend``: ``mj_step``, ``d->ctrl`` 에 PD 토크 (``<motor>`` 액추에이터)
  - 시임은 ``MotorCommand``(q_target/kp/kd/tau_ff)를 전달하고 PD 변환은 각 백엔드가 담당
  - ``robot_types.hpp``: 양측이 공유하는 타입·상수·관절 리밋
  - 실행 파일 분리: ``wholebody_rl``(실기) / ``wholebody_sim``(``-DG1_WITH_MUJOCO=ON``)

- 게임패드 부호 규약(``ly``, ``-lx``, ``-rx``)을 관측식에서 백엔드로 이동
- 주파수 모니터가 정책 실행 주기를 실제로 보고 (이전에는 항상 ``Control loop: 0 Hz``)

Added
~~~~~
- ``tools/fake_lowstate``: 로봇 없이 ``rt/lowstate`` 를 발행해 실기 경로 회귀 검증
  (CRC 계산, 게임패드 반복 탭으로 기동 지연에 무관)
- MuJoCo 백엔드 초기화: 접지 높이를 충돌 판정으로 탐색(0.793 → 0.786),
  정책 개입 전까지 부동 베이스 고정

Fixed
~~~~~
- ``state_`` 초기값이 ``RL_POLICY_ACTIVE`` 로 되어 있어 기본 자세 단계를 건너뛰던 문제
  → ``WAIT_FOR_INIT_COMMAND`` 로 복원
- ``GetObservation``/``Control`` 의 널 가드 복원 (주석 처리 시 DDS 미수신 상태에서 segfault)

2026-08-02
----------

Changed
~~~~~~~
- onnxruntime을 소스 빌드 대신 prebuilt 릴리스 tarball로 전환(1.11.0 → 1.28.0)

  - 최초 configure 때 ``thirdparty/`` 에 한 번만 받아 압축 해제, 이후 재사용
    (``rm -rf build`` 해도 다시 받지 않음)
  - SHA256 검증, x86_64/aarch64 자동 선택
  - 디렉터리 이름에 버전·아키텍처가 포함되어 버전 변경 시 옛 디렉터리를 재사용하지 않음
  - ``find_library`` + ``UNKNOWN IMPORTED``로 선언(tarball 동봉 CMake config는 경로가 깨져 있어 사용 불가)
  - 오프라인 빌드: tarball을 ``thirdparty/`` 에 직접 풀거나 ``-DONNXRUNTIME_DIR=`` 지정

- yaml-cpp 해석 순서 도입(vendored 소스 → 시스템 패키지 → 소스 다운로드), 정적 링크로 변경
- ``$ORIGIN`` RPATH 적용으로 실행 시 ``LD_LIBRARY_PATH`` 불필요

- ``thirdparty/onnxruntime`` 서브모듈 제거(더 이상 참조되지 않음, clone 용량 약 2.7GB 감소)

Added
~~~~~
- ``-DG1_WITH_MUJOCO=ON`` 옵션: MuJoCo prebuilt를 받아 ``mujoco::mujoco`` 타겟 제공(sim2sim 대비)
- ``tools/policy_check.cpp``: 정책 추론 결과(float 비트패턴 포함) 덤프 도구

  - 세션 설정은 ``wholebody_rl``과 동일, shape은 모델에서 읽어 어떤 정책에도 사용 가능
  - onnxruntime 버전 교체 / 정책 재export 시 회귀 비교용, ``-DG1_BUILD_TOOLS=OFF``로 비활성화

Fixed
~~~~~
- onnxruntime 1.12에서 제거된 ``GetInputName``/``GetOutputName`` 사용

  - ``ORT_API_VERSION`` 분기로 1.11 이하/1.12 이상 모두 빌드되도록 수정
  - 기존에 allocator가 할당한 io 이름 문자열을 해제하지 않던 누수도 함께 해결

- 검증: 1.11.0과 1.28.0에서 ``policy/basic_walk/policy.onnx`` 추론 결과 비교
  (36개 출력값, 최대 절대오차 7.2e-07 / 상대오차 8.0e-06 — float32 반올림 수준)


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

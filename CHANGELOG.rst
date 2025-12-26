Changelog
=========

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

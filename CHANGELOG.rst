Changelog
=========

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

cmake_minimum_required(VERSION 3.14)
project(g1_sim2real
  VERSION 1.0.0
  DESCRIPTION "Unitree G1 RL Sim2Real Deployment"
  LANGUAGES CXX
)

# ==== C++ Standard ====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "=== ${PROJECT_NAME} v${PROJECT_VERSION} ===")
message(STATUS "System Arch: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")

# ==== Options ====
# MuJoCo는 아직 이 저장소의 소스에서 쓰이지 않으므로 기본 OFF.
# sim2sim 타겟을 추가할 때 -DG1_WITH_MUJOCO=ON 으로 켜면 mujoco::mujoco 타겟이 생깁니다.
option(G1_WITH_MUJOCO "Fetch the MuJoCo prebuilt binary and expose the mujoco::mujoco target" OFF)
option(G1_BUILD_TOOLS "Build the helper tools in tools/" ON)

# 아래 두 버전을 바꾸면 해당 릴리스를 받아옵니다.
# 기본 버전이 아닐 경우 SHA256 검증은 자동으로 생략됩니다(해시가 버전마다 다르므로).
set(ONNXRUNTIME_VERSION 1.28.0 CACHE STRING "ONNX Runtime prebuilt release to download")
set(MUJOCO_VERSION      3.11.0 CACHE STRING "MuJoCo prebuilt release to download")

# 이미 풀어둔 prebuilt가 있으면 그 경로를 주세요. 다운로드를 아예 건너뜁니다(오프라인 빌드).
set(ONNXRUNTIME_DIR "" CACHE PATH "Existing prebuilt ONNX Runtime root; skips the download")
set(MUJOCO_DIR      "" CACHE PATH "Existing prebuilt MuJoCo root; skips the download")

set(ONNXRUNTIME_PINNED_VERSION 1.28.0)
set(MUJOCO_PINNED_VERSION      3.11.0)

# ==== PATHS ====
set(UNITREE_SDK2_DIR    ${CMAKE_SOURCE_DIR}/../unitree_sdk2)

set(THIRD_PARTY_DIR     ${CMAKE_SOURCE_DIR}/thirdparty)
set(YAML_CPP_SRC_DIR    ${THIRD_PARTY_DIR}/yaml-cpp)

# ==== Arch-dependent settings ====
# 주의: 두 프로젝트의 아키텍처 표기가 다릅니다 (ONNX Runtime: x64 / MuJoCo: x86_64)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  set(UNITREE_LIB_DIR    ${UNITREE_SDK2_DIR}/lib/x86_64)
  set(UNITREE_3P_LIB_DIR ${UNITREE_SDK2_DIR}/thirdparty/lib/x86_64)
  set(ORT_ARCH    x64)
  set(ORT_SHA256  a3e1b79d7bb1bf09696ce675f49e4064e6c81f6202b8225624fff0e93f8d6407)
  set(MJ_ARCH     x86_64)
  set(MJ_SHA256   681a9353b296a06682e9e757d6d21ab7a0a5a5bdb218a1c05cb9a38536817c79)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(UNITREE_LIB_DIR    ${UNITREE_SDK2_DIR}/lib/aarch64)
  set(UNITREE_3P_LIB_DIR ${UNITREE_SDK2_DIR}/thirdparty/lib/aarch64)
  set(ORT_ARCH    aarch64)
  set(ORT_SHA256  e15ff8b5d85afe6c144d97c6fd432254bf76a219daaf17658087d6ecb3e8f0bb)
  set(MJ_ARCH     aarch64)
  set(MJ_SHA256   572616d27117a9739fcc6affb8d21ff24a6b4d376a8dc4756f1602beb82f4e83)
else()
  message(FATAL_ERROR "Unsupported system arch: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

include(FetchContent)

# ==== Prebuilt tarball helper ====
# tarball을 thirdparty/ 아래에 풀어둡니다. build/ 를 지워도 남아 있으므로
# 다운로드는 최초 configure 때 딱 한 번만 일어납니다.
#
# DEST 이름은 tarball의 최상위 디렉터리 이름과 같게 잡습니다. 즉 직접 받아서
# thirdparty/ 에 풀어두면 CMake는 다운로드를 통째로 건너뜁니다(오프라인 빌드).
function(g1_fetch_prebuilt)
  cmake_parse_arguments(ARG "" "NAME;URL;SHA256;DEST" "" ${ARGN})

  if (IS_DIRECTORY "${ARG_DEST}")
    return()
  endif()

  get_filename_component(archive_name "${ARG_URL}" NAME)
  set(download_dir "${THIRD_PARTY_DIR}/.download")
  set(archive "${download_dir}/${archive_name}")
  set(staging "${download_dir}/${ARG_NAME}-staging")

  message(STATUS "${ARG_NAME}: downloading ${archive_name}")
  if (ARG_SHA256)
    file(DOWNLOAD "${ARG_URL}" "${archive}"
         EXPECTED_HASH SHA256=${ARG_SHA256} STATUS download_status)
  else()
    file(DOWNLOAD "${ARG_URL}" "${archive}" STATUS download_status)
  endif()
  list(GET download_status 0 download_code)
  if (NOT download_code EQUAL 0)
    list(GET download_status 1 download_message)
    file(REMOVE "${archive}")
    message(FATAL_ERROR
      "${ARG_NAME}: download failed - ${download_message}\n"
      "  ${ARG_URL}\n"
      "  오프라인이라면 tarball을 직접 받아 ${THIRD_PARTY_DIR} 에 풀어주세요.")
  endif()

  # 임시 폴더에 푼 뒤 옮깁니다. 중간에 실패해도 반쪽짜리 DEST가 남지 않습니다.
  file(REMOVE_RECURSE "${staging}")
  file(MAKE_DIRECTORY "${staging}")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E tar xzf "${archive}"
    WORKING_DIRECTORY "${staging}"
    RESULT_VARIABLE extract_code
  )
  if (NOT extract_code EQUAL 0)
    message(FATAL_ERROR "${ARG_NAME}: failed to extract ${archive_name}")
  endif()

  file(GLOB extracted "${staging}/*")
  list(LENGTH extracted extracted_count)
  if (NOT extracted_count EQUAL 1)
    message(FATAL_ERROR
      "${ARG_NAME}: expected one top-level directory in ${archive_name}, got ${extracted_count}")
  endif()
  file(RENAME "${extracted}" "${ARG_DEST}")

  file(REMOVE_RECURSE "${staging}")
  file(REMOVE "${archive}")
  file(GLOB download_leftovers "${download_dir}/*")
  if (NOT download_leftovers)
    file(REMOVE_RECURSE "${download_dir}")
  endif()
  message(STATUS "${ARG_NAME}: unpacked into ${ARG_DEST}")
endfunction()

# 핀 버전이 아니면 해시 검증을 건너뜁니다(해시는 버전·아키텍처마다 다르므로).
if (NOT ONNXRUNTIME_VERSION VERSION_EQUAL ONNXRUNTIME_PINNED_VERSION)
  set(ORT_SHA256 "")
  message(WARNING "ONNX Runtime ${ONNXRUNTIME_VERSION} is not the pinned version "
                  "(${ONNXRUNTIME_PINNED_VERSION}); skipping SHA256 verification.")
endif()
if (NOT MUJOCO_VERSION VERSION_EQUAL MUJOCO_PINNED_VERSION)
  set(MJ_SHA256 "")
  if (G1_WITH_MUJOCO)
    message(WARNING "MuJoCo ${MUJOCO_VERSION} is not the pinned version "
                    "(${MUJOCO_PINNED_VERSION}); skipping SHA256 verification.")
  endif()
endif()

# ==== ONNXRUNTIME (prebuilt) ====
# 소스 빌드 없이 릴리스 tarball(헤더 + .so)만 받아서 IMPORTED 타겟으로 연결합니다.
# 경로에 버전·아키텍처가 들어가므로 버전을 바꾸면 새로 받습니다(옛 디렉터리를 조용히 재사용하지 않음).
# ONNXRUNTIME_DIR 를 주면 다운로드 없이 그 경로를 그대로 씁니다.
set(ONNXRUNTIME_ROOT "${THIRD_PARTY_DIR}/onnxruntime-linux-${ORT_ARCH}-${ONNXRUNTIME_VERSION}")
if (ONNXRUNTIME_DIR)
  set(ONNXRUNTIME_ROOT "${ONNXRUNTIME_DIR}")
endif()

g1_fetch_prebuilt(
  NAME   onnxruntime
  URL    https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ORT_ARCH}-${ONNXRUNTIME_VERSION}.tgz
  SHA256 ${ORT_SHA256}
  DEST   ${ONNXRUNTIME_ROOT}
)

# 확장자(.so/.dylib)와 lib 접두사는 CMake가 플랫폼 규칙대로 붙여 찾습니다.
find_library(ONNXRUNTIME_LIB
  NAMES onnxruntime
  HINTS ${ONNXRUNTIME_ROOT}/lib
  NO_DEFAULT_PATH
)
if (NOT ONNXRUNTIME_LIB)
  message(FATAL_ERROR "onnxruntime library not found in: ${ONNXRUNTIME_ROOT}/lib")
endif()
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIB}")

# 정적/동적 여부를 단정하지 않는 UNKNOWN IMPORTED 사용
# (릴리스 tarball에 lib/cmake/onnxruntime/ 가 들어있지만, 그 config는 존재하지 않는
#  include/onnxruntime 및 lib64 경로를 참조해 find_package가 실패합니다. 직접 선언합니다.)
add_library(onnxruntime UNKNOWN IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION             "${ONNXRUNTIME_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
)

# ==== MUJOCO (prebuilt, optional) ====
if (G1_WITH_MUJOCO)
  # MuJoCo 릴리스 태그에는 'v' 접두사가 없고, tarball 최상위는 mujoco-<ver> 입니다.
  set(MUJOCO_ROOT "${THIRD_PARTY_DIR}/mujoco-${MUJOCO_VERSION}")
  if (MUJOCO_DIR)
    set(MUJOCO_ROOT "${MUJOCO_DIR}")
  endif()

  g1_fetch_prebuilt(
    NAME   mujoco
    URL    https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-${MJ_ARCH}.tar.gz
    SHA256 ${MJ_SHA256}
    DEST   ${MUJOCO_ROOT}
  )

  find_library(MUJOCO_LIB
    NAMES mujoco
    HINTS ${MUJOCO_ROOT}/lib
    NO_DEFAULT_PATH
  )
  if (NOT MUJOCO_LIB)
    message(FATAL_ERROR "mujoco library not found in: ${MUJOCO_ROOT}/lib")
  endif()
  message(STATUS "MuJoCo: ${MUJOCO_LIB}")

  add_library(mujoco::mujoco UNKNOWN IMPORTED GLOBAL)
  set_target_properties(mujoco::mujoco PROPERTIES
    IMPORTED_LOCATION             "${MUJOCO_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_ROOT}/include"
  )
  # 뷰어(mjr_* 렌더링)를 직접 쓰려면: sudo apt install libglfw3-dev
  # 미리 빌드된 뷰어는 ${MUJOCO_ROOT}/bin/simulate 에 있습니다.
endif()

# ==== YAML-CPP ====
# yaml-cpp는 공식 prebuilt 바이너리 릴리스가 없습니다.
# vendored 소스 > 시스템 패키지(libyaml-cpp-dev) > 소스 다운로드 순으로 해결합니다.
set(YAML_CPP_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS   OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
# 정적 링크: 로봇에 배포할 때 같이 챙겨야 할 .so가 하나 줄어듭니다.
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "Build yaml-cpp as a shared library")

if (EXISTS "${YAML_CPP_SRC_DIR}/CMakeLists.txt")
  message(STATUS "yaml-cpp: vendored source (${YAML_CPP_SRC_DIR})")
  add_subdirectory(${YAML_CPP_SRC_DIR})
else()
  find_package(yaml-cpp QUIET)
  if (yaml-cpp_FOUND)
    message(STATUS "yaml-cpp: system package ${yaml-cpp_VERSION}")
  else()
    message(STATUS "yaml-cpp: not found locally, fetching source 0.8.0")
    FetchContent_Declare(yaml_cpp
      GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
      GIT_TAG        0.8.0
      GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(yaml_cpp)
  endif()
endif()

# 0.8+ 는 yaml-cpp::yaml-cpp, apt로 설치되는 0.6/0.7 은 yaml-cpp 를 노출합니다.
if (TARGET yaml-cpp::yaml-cpp)
  set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
else()
  set(YAML_CPP_TARGET yaml-cpp)
endif()

# ==== Common Deps ====
add_library(com_deps INTERFACE)
target_include_directories(com_deps INTERFACE
  ${CMAKE_SOURCE_DIR}/include
  ${UNITREE_SDK2_DIR}/include
  ${UNITREE_SDK2_DIR}/thirdparty/include
  ${UNITREE_SDK2_DIR}/thirdparty/include/ddscxx
)
target_link_directories(com_deps INTERFACE
  ${UNITREE_LIB_DIR}
  ${UNITREE_3P_LIB_DIR}
)
target_link_libraries(com_deps INTERFACE
  unitree_sdk2
  ddsc
  ddscxx
  pthread
)

# ==== Libs ====
add_library(utils STATIC lib/utils.cpp)
target_include_directories(utils PUBLIC ${CMAKE_SOURCE_DIR}/include)

# ==== Libs ====
# 공통 제어 로직: DDS도 MuJoCo도 모릅니다 (read/control/write 시임만 노출)
add_library(wholebody_control STATIC src/wholebody_rl.cpp)
target_include_directories(wholebody_control PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(wholebody_control PUBLIC utils onnxruntime ${YAML_CPP_TARGET})

# ==== Executables ====
add_executable(g1_arm_example src/g1_arm_example.cpp)

# 실기: 공통 제어 + DDS 백엔드
add_executable(wholebody_rl src/main_real.cpp src/unitree_comm.cpp)

# ==== Link ====
target_link_libraries(g1_arm_example PRIVATE com_deps)
target_link_libraries(wholebody_rl  PRIVATE wholebody_control com_deps)

# 시뮬: 공통 제어 + MuJoCo (-DG1_WITH_MUJOCO=ON 일 때만)
if (G1_WITH_MUJOCO)
  # 뷰어를 직접 만들었으므로 GLFW + OpenGL 만 필요합니다. MuJoCo 의 simulate 소스를
  # 끌어오지 않으니 lodepng 도 필요 없습니다.
  #   sudo apt install libglfw3-dev
  find_package(OpenGL REQUIRED)
  find_package(glfw3 QUIET)
  if (NOT glfw3_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GLFW3 REQUIRED glfw3)
  endif()

  add_executable(wholebody_sim
    src/main_sim.cpp
    src/mujoco_sim.cpp
    src/mujoco_viewer.cpp
  )
  target_link_libraries(wholebody_sim PRIVATE wholebody_control mujoco::mujoco OpenGL::GL)
  if (glfw3_FOUND)
    target_link_libraries(wholebody_sim PRIVATE glfw)
  else()
    target_include_directories(wholebody_sim PRIVATE ${GLFW3_INCLUDE_DIRS})
    target_link_directories(wholebody_sim PRIVATE ${GLFW3_LIBRARY_DIRS})
    target_link_libraries(wholebody_sim PRIVATE ${GLFW3_LIBRARIES})
  endif()
  set_target_properties(wholebody_sim PROPERTIES
    BUILD_RPATH "$ORIGIN;${ONNXRUNTIME_ROOT}/lib;${MUJOCO_ROOT}/lib"
  )
endif()

# ==== Tools ====
# policy_check: 정책 추론 결과를 덤프해 onnxruntime 버전 간/정책 간 회귀를 비교
if (G1_BUILD_TOOLS)
  add_executable(policy_check tools/policy_check.cpp)
  target_link_libraries(policy_check PRIVATE onnxruntime)

  # fake_lowstate: 로봇 대신 rt/lowstate 를 발행해 로봇 없이 wholebody_rl 을 돌려봅니다
  add_executable(fake_lowstate tools/fake_lowstate.cpp)
  target_link_libraries(fake_lowstate PRIVATE utils com_deps)
endif()

# ==== RPATH ====
# onnxruntime은 동적 라이브러리로만 배포되므로 실행 시 .so를 찾을 경로가 필요합니다.
# $ORIGIN = 실행 파일이 있는 디렉터리. 실행 파일 옆이나 ../lib 에 .so를 복사해두면
# 폴더째 로봇으로 옮겨도 LD_LIBRARY_PATH 없이 동작합니다.
set(ORT_RPATH "$ORIGIN;${ONNXRUNTIME_ROOT}/lib")
set_target_properties(wholebody_rl PROPERTIES
  BUILD_RPATH   "${ORT_RPATH}"
  INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib"
)
if (G1_BUILD_TOOLS)
  set_target_properties(policy_check PROPERTIES BUILD_RPATH "${ORT_RPATH}")
endif()

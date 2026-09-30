cmake_minimum_required(VERSION 3.10)
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

# ==== PATHS ====
set(UNITREE_SDK2_DIR    ${CMAKE_SOURCE_DIR}/../unitree_sdk2)

set(THIRD_PARTY_DIR     ${CMAKE_SOURCE_DIR}/thirdparty)
set(YAML_CPP_SRC_DIR    ${THIRD_PARTY_DIR}/yaml-cpp)

# ONNX Runtime >= 1.14 is required: the policy is exported with opset 18, and the loader uses
# GetInputNameAllocated, which replaced GetInputName in 1.14.
#
# Searched in order, first hit wins. Override with -DONNXRUNTIME_ROOT=<path> to point anywhere else
# (a prebuilt release directory, or thirdparty/onnxruntime built from source into build/Release):
#   1. a prebuilt release unpacked into thirdparty/ -- run scripts/get_onnxruntime.sh to fetch the
#      one matching this machine's architecture
#   2. thirdparty/onnxruntime built from source (the submodule; slow, see the README)
if (NOT ONNXRUNTIME_ROOT)
  file(GLOB ORT_CANDIDATES ${THIRD_PARTY_DIR}/onnxruntime-linux-*)
  list(APPEND ORT_CANDIDATES ${THIRD_PARTY_DIR}/onnxruntime)
  foreach(ORT_DIR ${ORT_CANDIDATES})
    if (EXISTS "${ORT_DIR}/include/onnxruntime_cxx_api.h" OR
        EXISTS "${ORT_DIR}/include/onnxruntime/core/session/onnxruntime_cxx_api.h")
      set(ONNXRUNTIME_ROOT ${ORT_DIR} CACHE PATH "ONNX Runtime root")
      break()
    endif()
  endforeach()
endif()
if (NOT ONNXRUNTIME_ROOT)
  message(FATAL_ERROR
    "ONNX Runtime not found. Run scripts/get_onnxruntime.sh, or pass -DONNXRUNTIME_ROOT=<path>. "
    "Looked in: ${ORT_CANDIDATES}")
endif()

# ==== Unitree SDK2 ====
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(UNITREE_LIB_DIR    ${UNITREE_SDK2_DIR}/lib/x86_64)
  set(UNITREE_3P_LIB_DIR ${UNITREE_SDK2_DIR}/thirdparty/lib/x86_64)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(UNITREE_LIB_DIR    ${UNITREE_SDK2_DIR}/lib/aarch64)
  set(UNITREE_3P_LIB_DIR ${UNITREE_SDK2_DIR}/thirdparty/lib/aarch64)
else()
  message(FATAL_ERROR "Unsupported system arch: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# ==== YAML-CPP ====
set(YAML_CPP_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS   OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)

if (NOT EXISTS "${YAML_CPP_SRC_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "yaml-cpp not found: ${YAML_CPP_SRC_DIR}")
endif()
add_subdirectory(${YAML_CPP_SRC_DIR})

# ==== ONNXRUNTIME ====
find_library(ONNXRUNTIME_LIB
  NAMES onnxruntime libonnxruntime.so.1.22.0 libonnxruntime.so.1
  HINTS ${ONNXRUNTIME_ROOT}/lib ${ONNXRUNTIME_ROOT}/build/Release
  NO_DEFAULT_PATH
)
if (NOT ONNXRUNTIME_LIB)
  message(FATAL_ERROR "onnxruntime not found under: ${ONNXRUNTIME_ROOT} (lib/ or build/Release/)")
endif()
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIB}")

# Fail here rather than deep in a compile error: 1.14 introduced GetInputNameAllocated and the
# opset 18 support the exported policies need. ORT_API_VERSION tracks the minor version (1.22 -> 22).
find_file(ONNXRUNTIME_C_API_HEADER onnxruntime_c_api.h
  PATHS ${ONNXRUNTIME_ROOT}/include ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session
  NO_DEFAULT_PATH)
if (ONNXRUNTIME_C_API_HEADER)
  file(STRINGS ${ONNXRUNTIME_C_API_HEADER} ORT_API_VERSION_LINE REGEX "^#define ORT_API_VERSION ")
  string(REGEX MATCH "[0-9]+" ORT_API_VERSION "${ORT_API_VERSION_LINE}")
  if (ORT_API_VERSION AND ORT_API_VERSION LESS 14)
    message(FATAL_ERROR
      "ONNX Runtime 1.${ORT_API_VERSION} at ${ONNXRUNTIME_ROOT} is too old; 1.14 or newer is "
      "required. Run scripts/get_onnxruntime.sh to fetch a current release.")
  endif()
  message(STATUS "ONNX Runtime API version: ${ORT_API_VERSION} (1.${ORT_API_VERSION}.x)")
endif()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
)
target_include_directories(onnxruntime INTERFACE "${ONNXRUNTIME_ROOT}/include")
if (EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session")  # source-tree build layout only
  target_include_directories(onnxruntime INTERFACE "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session")
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

# ==== Executables ====
add_executable(g1_arm_example src/g1_arm_example.cpp)
add_executable(wholebody_rl  src/wholebody_rl.cpp)

# ==== Link ====
target_link_libraries(g1_arm_example PRIVATE com_deps)
target_link_libraries(wholebody_rl  PRIVATE utils com_deps onnxruntime yaml-cpp)

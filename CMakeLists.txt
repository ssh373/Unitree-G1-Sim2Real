cmake_minimum_required(VERSION 3.10)
project(g1_sim2real LANGUAGES CXX)

# ==== C++ ====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "System Arch: ${CMAKE_SYSTEM_PROCESSOR}")

# ==== PATHS ====
set(UNITREE_SDK2_DIR    ${CMAKE_SOURCE_DIR}/../unitree_sdk2)

set(THIRD_PARTY_DIR     ${CMAKE_SOURCE_DIR}/thirdparty)
set(YAML_CPP_SRC_DIR    ${THIRD_PARTY_DIR}/yaml-cpp)
set(ONNXRUNTIME_ROOT    ${THIRD_PARTY_DIR}/onnxruntime)
set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_ROOT}/build/Release)

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
  NAMES onnxruntime
  HINTS ${ONNXRUNTIME_LIB_DIR}
  NO_DEFAULT_PATH
)
if (NOT ONNXRUNTIME_LIB)
  message(FATAL_ERROR "onnxruntime not found in: ${ONNXRUNTIME_LIB_DIR}")
endif()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
)
target_include_directories(onnxruntime INTERFACE
  "${ONNXRUNTIME_ROOT}/include"
  "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session"
)

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

cmake_minimum_required(VERSION 3.10)
project(arc_g1)

# c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# detect architecture
message(STATUS "Detected architecture: ${CMAKE_SYSTEM_PROCESSOR}")

# choose the correct lib path based on architecture
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(UNITREE_LIB_DIR "${CMAKE_SOURCE_DIR}/../unitree_sdk2-main/lib/x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(UNITREE_LIB_DIR "${CMAKE_SOURCE_DIR}/../unitree_sdk2-main/lib/aarch64")
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# ONNXRUNTIME configuration
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime")

# YAML configuration
set(YAML_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/yaml-cpp)
add_library(yaml-cpp SHARED IMPORTED)
set_target_properties(yaml-cpp PROPERTIES
  IMPORTED_LOCATION ${YAML_CPP_DIR}/build/libyaml-cpp.so
  INTERFACE_INCLUDE_DIRECTORIES ${YAML_CPP_DIR}/include
)

# include path
include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/../unitree_sdk2-main/include
  ${CMAKE_SOURCE_DIR}/../unitree_sdk2-main/thirdparty/include
  ${CMAKE_SOURCE_DIR}/../unitree_sdk2-main/thirdparty/include/ddscxx
  ${ONNXRUNTIME_ROOT}/include
  ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session
  ${YAML_CPP_DIR}/include
)

# link path
link_directories(
  ${UNITREE_LIB_DIR}
  ${CMAKE_SOURCE_DIR}/lib
  ${ONNXRUNTIME_ROOT}/build_jetson/Release
  # ${CMAKE_SOURCE_DIR}/../unitree_sdk2-main/thirdparty/lib/x86_64
)

add_library(utils
  lib/utils.cpp
)

# arc_g1 source
add_executable(g1_arm_example src/g1_arm_example.cpp)
add_executable(wholebody_rl src/wholebody_rl.cpp)
add_executable(wholebody_arc src/wholebody_arc.cpp)
# add_executable(wholebody_arc_29dof src/wholebody_arc_29dof.cpp)
add_executable(wholebody_korea src/wholebody_korea.cpp)
add_executable(wholebody_korea_inb src/wholebody_korea_inb.cpp)

# link with the sdk library
target_link_libraries(g1_arm_example
  unitree_sdk2
  ddsc
  ddscxx
  pthread
)

target_link_libraries(wholebody_rl
  utils
  yaml-cpp
  unitree_sdk2
  ddsc
  ddscxx
  pthread
  onnxruntime
)

target_link_libraries(wholebody_arc
  utils
  yaml-cpp
  unitree_sdk2
  ddsc
  ddscxx
  pthread
  onnxruntime
)

# target_link_libraries(wholebody_arc_29dof
#   utils
#   yaml-cpp
#   unitree_sdk2
#   ddsc
#   ddscxx
#   pthread
#   onnxruntime
# )

target_link_libraries(wholebody_korea
  utils
  yaml-cpp
  unitree_sdk2
  ddsc
  ddscxx
  pthread
  onnxruntime
)

target_link_libraries(wholebody_korea_inb
  utils
  yaml-cpp
  unitree_sdk2
  ddsc
  ddscxx
  pthread
  onnxruntime
)

/*****************************************************************************
** Loads a policy .onnx exactly the way wholebody_rl.cpp does, feeds a set of
** deterministic observations and dumps the raw output bits.
**
** Use it to tell whether something actually changed:
**   - after bumping ONNXRUNTIME_VERSION  -> diff the output of two builds
**   - after re-exporting a policy        -> diff against the previous policy
**
** Bit patterns are printed alongside the decimal values because two
** onnxruntime versions rarely agree bit-for-bit (the Gemm kernels accumulate
** in a different order), and eyeballing decimals hides that.
**
**   ./policy_check ../policy/basic_walk/policy.onnx
*****************************************************************************/
#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Product of the declared dims, with dynamic dims (-1) treated as 1.
size_t ElementCount(const std::vector<int64_t>& shape)
{
  size_t n = 1;
  for (int64_t d : shape) n *= (d > 0 ? static_cast<size_t>(d) : 1);
  return n;
}

std::string ShapeToString(const std::vector<int64_t>& shape)
{
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i)
  {
    s += (i ? ", " : "") + std::to_string(shape[i]);
  }
  return s + "]";
}

// Deterministic, reproducible across machines: no RNG, no time dependency.
float SyntheticObs(int obs_case, size_t i)
{
  switch (obs_case)
  {
    case 0:  return 0.0f;
    case 1:  return std::sin(static_cast<float>(i) * 0.37f) * 0.5f;
    default: return (i % 2 ? -1.0f : 1.0f) * 0.01f * static_cast<float>(i);
  }
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc < 2)
  {
    std::fprintf(stderr, "usage: %s <policy.onnx>\n", argv[0]);
    return 1;
  }
  const char* model_path = argv[1];

  std::printf("onnxruntime : header ORT_API_VERSION=%d, runtime %s\n",
              ORT_API_VERSION, OrtGetApiBase()->GetVersionString());
  std::printf("model       : %s\n", model_path);

  // Mirror the session setup in WholeBodyRL::LoadOnnxModel()
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_cpu_RL");

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  Ort::Session session(env, model_path, session_options);
  Ort::AllocatorWithDefaultOptions allocator;

  std::string input_name = session.GetInputNameAllocated(0, allocator).get();
  std::string output_name = session.GetOutputNameAllocated(0, allocator).get();

  // Read the shapes off the model so this works for any policy.
  // Keep the TypeInfo alive: the shape info borrows from it.
  Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
  std::vector<int64_t> input_shape = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
  Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
  std::vector<int64_t> output_shape = output_type_info.GetTensorTypeAndShapeInfo().GetShape();

  for (int64_t& d : input_shape) if (d < 0) d = 1;   // pin dynamic batch to 1

  const size_t num_obs = ElementCount(input_shape);
  std::printf("input       : %s %s\n", input_name.c_str(), ShapeToString(input_shape).c_str());
  std::printf("output      : %s %s\n", output_name.c_str(), ShapeToString(output_shape).c_str());

  const char* input_names[] = {input_name.c_str()};
  const char* output_names[] = {output_name.c_str()};

  for (int obs_case = 0; obs_case < 3; ++obs_case)
  {
    std::vector<float> obs(num_obs);
    for (size_t i = 0; i < num_obs; ++i) obs[i] = SyntheticObs(obs_case, i);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, obs.data(), obs.size(), input_shape.data(), input_shape.size());

    std::vector<Ort::Value> outputs = session.Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    const size_t n = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    const float* actions = outputs[0].GetTensorData<float>();

    std::printf("case%d bits :", obs_case);
    for (size_t i = 0; i < n; ++i)
    {
      uint32_t bits;
      std::memcpy(&bits, &actions[i], sizeof(bits));
      std::printf(" %08x", bits);
    }
    std::printf("\ncase%d val  :", obs_case);
    for (size_t i = 0; i < n; ++i) std::printf(" % .9g", actions[i]);
    std::printf("\n");
  }

  return 0;
}

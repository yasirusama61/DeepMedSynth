// infer_unet_tensorrt.cpp
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>

int main() {
    std::ifstream file("unet_model.trt", std::ios::binary);
    if (!file) { std::cerr << "Engine file not found!\n"; return -1; }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), size, nullptr);

    if (engine) std::cout << "✅ TensorRT engine loaded in C++\n";
    else std::cout << "❌ Failed to load engine\n";

    engine->destroy();
    runtime->destroy();
    return 0;
}

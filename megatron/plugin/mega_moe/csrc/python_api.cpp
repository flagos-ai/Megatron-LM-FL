// Copyright (c) FlagOS Team, BAAI Corporation.
#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "apis/sm90_mega.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _mega_moe_C
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SM90 MegaMoE C++ extension for Megatron-LM";
    deep_gemm::mega::register_sm90_apis(m);
}

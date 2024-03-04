// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gflags/gflags.h>

#include "openvino/openvino.hpp"
#include "openvino/pass/convert_i64_to_i32.hpp"
#include "openvino/pass/validate.hpp"
#include "openvino/pass/manager.hpp"

DEFINE_string(m, "", "OpenVINO model (.xml) to convert");
DEFINE_string(o, "", "Output OpenVINO model (.xml) file");

namespace {

bool parseCommandLine(int* argc, char*** argv) {
    gflags::SetUsageMessage("Usage: i64toi32 -m FILE -o FILE");
    gflags::ParseCommandLineFlags(argc, argv, true);
    if (FLAGS_m.empty()) {
        throw std::invalid_argument("Path to model xml file is required");
    }
    if (FLAGS_o.empty()) {
        throw std::invalid_argument("Path to output model xml file is requried");
    }
    gflags::ShutDownCommandLineFlags();
    return true;
}

} // namespace

int main(int argc, char* argv[]) {
    if (!parseCommandLine(&argc, &argv)) {
        return -1;
    }

    ov::Core core;
    auto model = core.read_model(FLAGS_m);

    ov::pass::Manager m;
    m.register_pass<ov::pass::ConvertI64ToI32>();
    m.register_pass<ov::pass::Validate>();
    m.run_passes(model);

    ov::save_model(model, FLAGS_o);
    return 0;
}

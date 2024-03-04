// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/convert_i64_to_i32.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/convert_precision.hpp"

bool ov::pass::ConvertI64ToI32::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(ConvertI64ToI32);
    ov::pass::Manager m(get_pass_config());
    m.register_pass<ov::pass::ConvertPrecision>(precisions_map{{ov::element::i64, ov::element::i32}});
    m.run_passes(f);
    return false;
}

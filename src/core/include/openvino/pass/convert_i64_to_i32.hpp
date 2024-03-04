// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace pass {
/**
 * @brief ConvertI64ToI32 transformation
 * @ingroup ov_pass_cpp_api
 */
class OPENVINO_API ConvertI64ToI32 : public ModelPass {
public:
    OPENVINO_RTTI("ConvertFP64ToFPI32");
    bool run_on_model(const std::shared_ptr<ov::Model>&) override;
};
}  // namespace pass
}  // namespace ov

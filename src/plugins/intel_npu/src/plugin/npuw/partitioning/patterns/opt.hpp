// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <string>

#include "openvino/openvino.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace npuw {

// Model optimization patterns. Triggered by the plugin at the very top
namespace patterns {
namespace opt {

class DQMatMulCWi : public ov::pass::MatcherPass {
public:
    DQMatMulCWi();
};

struct Context {
    std::string pmm_dims;

    using PPtr = std::shared_ptr<ov::op::v0::Parameter>;
    using NPtr = std::shared_ptr<ov::Node>;

    using Axes = std::vector<std::size_t>;
    std::map<PPtr, Axes> closures_to_permute;
    void permute(PPtr orig_param, const Axes& order);

    std::set<PPtr> closures_to_f16;
    void to_f16(PPtr orig_param);

    using O = ov::Output<ov::Node>;
    struct DQParMM {
        PPtr w, s;
        NPtr mm;
    };
    using DQParMMs = std::vector<DQParMM>;
    std::map<std::pair<O, std::size_t>, DQParMMs> par_dq_mms;
    void register_parallel_matmul(O multiply, std::size_t axis, DQParMM&& mm);

    std::map<PPtr, std::pair<ov::ParameterVector, std::size_t>> params_to_concat;
    PPtr concat(ov::ParameterVector&& v, std::size_t dim);

    using Ref = std::reference_wrapper<Context>;
};

class DQMatMulCWu : public ov::pass::MatcherPass {
public:
    DQMatMulCWu();
};

class DQMatMulGQi : public ov::pass::MatcherPass {
public:
    explicit DQMatMulGQi(Context::Ref ctx);
};

class DQMatMulGQ2i : public ov::pass::MatcherPass {
public:
    explicit DQMatMulGQ2i(Context::Ref ctx);
};

class DQParMMGQ : public ov::pass::MatcherPass {
public:
    explicit DQParMMGQ(Context::Ref ctx);
};

void mergeParallelMatMuls(const std::shared_ptr<ov::Model>& m, Context& ctx);

class DQGather : public ov::pass::MatcherPass {
public:
    DQGather();
};

}  // namespace opt
}  // namespace patterns
}  // namespace npuw
}  // namespace ov

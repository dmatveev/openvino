// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>

#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace npuw {
namespace weights {

struct ZeroAllocator final {
    std::shared_ptr<ov::IRemoteContext> m_remote_ctx = nullptr;

    explicit ZeroAllocator(std::shared_ptr<ov::IRemoteContext> ctx) : m_remote_ctx(ctx) {
    }

    void* allocate(const size_t bytes, const size_t alignment);
    void deallocate(void * handle, const size_t /*bytes*/, size_t /*alignment*/);

    bool is_equal(const ZeroAllocator &other) const {
        return m_remote_ctx == other.m_remote_ctx;
    }
};

class Bank {
public:
    explicit Bank(const std::shared_ptr<const ov::ICore>& core)
        : m_core(core)
        , m_allocator(std::make_shared<ZeroAllocator>(m_core->get_default_context("NPU")._ptr)) {
    }

    // Capture CPU version of the tensor
    ov::Tensor update(const ov::Tensor& tensor);

    // Based on previously captured tensor allocate a new tensor (if needed) on a specified device
    ov::Tensor get(const ov::Tensor& tensor, const std::string& device);

private:
    // Default CPU bank. Filled by update()
    std::unordered_map<void*, ov::Tensor> m_bank;
    // Bank for specified device and their allocated memory
    std::unordered_map<std::string, std::unordered_map<void*, ov::Tensor>> m_device_bank;
    std::mutex m_mutex;
    std::shared_ptr<const ov::ICore> m_core = nullptr;
    std::shared_ptr<ZeroAllocator> m_allocator; // TODO: make it per-device
};

std::shared_ptr<Bank> bank(const std::string& bank_name, const std::shared_ptr<const ov::ICore>& core);

}  // namespace weights
}  // namespace npuw
}  // namespace ov

// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_api.h"
#include "logging.hpp"

#include "openvino/runtime/allocator.hpp"
#include "remote_context.hpp"

#include "weights_bank.hpp"


using ov::npuw::weights::Bank;

namespace {

class BankManager {
public:
    static BankManager& getInstance() {
        static BankManager instance;
        return instance;
    }

private:
    BankManager() {}
    BankManager(BankManager const&) = delete;
    void operator=(BankManager const&) = delete;

public:
    // Public API
    std::shared_ptr<Bank> getBank(const std::string& bank_name, const std::shared_ptr<const ov::ICore>& core);

private:
    // Data
    std::unordered_map<std::string, std::weak_ptr<Bank>> m_bank_map;
    std::mutex m_mutex;
};

} // anonymous namespace

void* ov::npuw::weights::ZeroAllocator::allocate(const size_t bytes, const size_t alignment) {
    auto pRC = std::dynamic_pointer_cast<::intel_npu::RemoteContextImpl>(m_remote_ctx);
    NPUW_ASSERT(pRC);

    ze_host_mem_alloc_desc_t hostMemDesc = { ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED };
    void* pHostMem = NULL;

#define ALIGN_UP(v, a) (((v)+(a)-1) & (~((a)-1)))
    std::size_t size = ALIGN_UP(bytes, alignment);
#undef ALIGN_UP

    ze_context_handle_t context = static_cast<ze_context_handle_t>(pRC->native_context());
    auto ret = zeMemAllocHost(context, &hostMemDesc, size, alignment, &pHostMem);
    NPUW_ASSERT(ret == ZE_RESULT_SUCCESS);
    return pHostMem;
}

void ov::npuw::weights::ZeroAllocator::deallocate(void * handle, const size_t /*bytes*/, size_t /*alignment*/) {
    auto pRC = std::dynamic_pointer_cast<::intel_npu::RemoteContextImpl>(m_remote_ctx);
    NPUW_ASSERT(pRC);

    ze_context_handle_t context = static_cast<ze_context_handle_t>(pRC->native_context());

    if (handle) {
        zeMemFree(context, handle);
    }
}

ov::Tensor Bank::update(const ov::Tensor& tensor) {
    if (!tensor) {
        OPENVINO_THROW("Uninitialized tensor in weights bank allocation!");
    }

    std::lock_guard<std::mutex> guard(m_mutex);
    m_bank[tensor.data()] = tensor;
    return tensor;
}

ov::Tensor Bank::get(const ov::Tensor& tensor, const std::string& device) {
    if (!tensor) {
        OPENVINO_THROW("Uninitialized tensor in weights bank allocation!");
    }

    if (device != "CPU" && device != "NPU") {
        OPENVINO_THROW("Unsupported device in weights bank allocation: ", device);
    }

    std::lock_guard<std::mutex> guard(m_mutex);

    auto iter_cpu = m_bank.find(tensor.data());
    if (iter_cpu == m_bank.end()) {
        OPENVINO_THROW("Unknown tensor in weights bank allocation!");
    }

    // If target device is CPU - just reuse the default bank.
    if (device == "CPU") {
        return iter_cpu->second;
    }

    // Non-CPU - check if the tensor is already there
    auto& device_bank = m_device_bank[device];
    auto iter_device = device_bank.find(tensor.data());
    if (iter_device != device_bank.end()) {
        // Already allocated on the device - reuse
        return iter_device->second;
    }

    // Allocation needed. Still device if allocate it on device or not.
    // auto remote_ctx = m_core->get_default_context(device)._ptr;
    // auto remote_tensor = remote_ctx->create_host_tensor(tensor.get_element_type(),
    //                                                     tensor.get_shape());
    auto allocated_tensor = ov::Tensor(tensor.get_element_type(),
                                       tensor.get_shape(),
                                       *m_allocator);
    tensor.copy_to(allocated_tensor);

    return (device_bank[tensor.data()] = allocated_tensor);
}

std::shared_ptr<Bank> BankManager::getBank(const std::string& bank_name, const std::shared_ptr<const ov::ICore>& core) {
    std::lock_guard<std::mutex> guard(m_mutex);

    auto iter = m_bank_map.find(bank_name);
    if (iter == m_bank_map.end()) {
        auto bank = std::make_shared<Bank>(core);
        m_bank_map[bank_name] = bank;
        return bank;
    }
    return iter->second.lock();
}

std::shared_ptr<Bank> ov::npuw::weights::bank(const std::string& bank_name,
                                              const std::shared_ptr<const ov::ICore>& core) {
    if (bank_name.empty()) {
        // Don't share this bank in manager
        return std::make_shared<Bank>(core);
    }

    auto& instance = BankManager::getInstance();
    return instance.getBank(bank_name, core);
}

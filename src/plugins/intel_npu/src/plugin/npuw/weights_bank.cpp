// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weights_bank.hpp"
#include "logging.hpp"

using ov::npuw::weights::Bank;

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

ov::Tensor Bank::update(const ov::Tensor& tensor) {
    if (!tensor) {
        OPENVINO_THROW("Uninitialized tensor in weights bank allocation!");
    }

    std::lock_guard<std::mutex> guard(m_mutex);
    m_bank[tensor.data()] = tensor;
    m_bytes.total_registered += tensor.get_byte_size();
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
    const auto tbytes = tensor.get_byte_size();
    m_bytes.pinned_total += tbytes;

    if (decide_for_dev(tensor, device)) {
        m_remote_ctx = m_core->get_default_context(device)._ptr;
        auto remote_tensor = m_remote_ctx->create_host_tensor(tensor.get_element_type(),
                                                              tensor.get_shape());
        auto allocated_tensor = ov::make_tensor(remote_tensor);
        tensor.copy_to(allocated_tensor);

        m_bytes.pinned_dev += tbytes;
        return (device_bank[tensor.data()] = allocated_tensor);
    }
    // ..Not ok - still return a cpu tensor. Store it in the device_bank
    // instead of the remote one to avoid duplicated alocations
    return (device_bank[tensor.data()] = iter_cpu->second);
}

bool Bank::decide_for_dev(const ov::Tensor &t, const std::string &) {
    // Called in context of allocation on the device.
    // Decide if this particular tensor should map to device or not.

    // Let the limit be hardcoded for now - to 1.7 GB
    const std::size_t DEV_LIMIT = 1700 * 1024 * 1024;

    NPUW_ASSERT(m_bytes.total_registered != 0);
    const std::size_t tbytes = t.get_byte_size();

    if (DEV_LIMIT >= m_bytes.total_registered) {
        // All allocations fit the limit - so it is an easy check
        return true;
    }
    if (m_bytes.pinned_dev == 0 && tbytes <= DEV_LIMIT) {
        // First allocation - let it be
        return true;
    }

    // Ratio is required to maintain the device tensor distribution across the network
    const float target_ratio = static_cast<float>(DEV_LIMIT) / m_bytes.total_registered;
    const float tobe_ratio = static_cast<float>(m_bytes.pinned_dev + tbytes) / m_bytes.pinned_total;

    if (tobe_ratio <= target_ratio) {
        // The resulting ratio is within treshold - let it be on device
        return true;
    } else {
        // The resulting ratio is above threshold - keep it to CPU
        return false;
    }
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

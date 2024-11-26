// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/aligned_buffer.hpp"

#include <iostream>
#include <iomanip>
#include <memory>

namespace ov {
class MappedMemory;
struct AlignedBuffer;
}
namespace {
template<class T>
struct P {
    static void p_alloc(const char *data, std::size_t size) {}
    static void p_dealloc(const char *data, std::size_t size) {}
};

template<>
struct P<std::shared_ptr<ov::MappedMemory> > {
    static void p_alloc(const char *data, std::size_t size) {
        std::cout << "Constructing shared buffer: data=" << static_cast<const void*>(data) << ", size=" << size << std::endl;
    }
    static void p_dealloc(const char *data, std::size_t size) {
        std::cout << "Destructing shared buffer: data=" << static_cast<const void*>(data) << ", size=" << size << std::endl;
    }
};
}

namespace ov {

/// \brief SharedBuffer class to store pointer to pre-allocated buffer. Own the shared object.
template <typename T>
class SharedBuffer : public ov::AlignedBuffer {
public:
    SharedBuffer(char* data, size_t size, const T& shared_object) : _shared_object(shared_object) {
        m_allocated_buffer = data;
        m_aligned_buffer = data;
        m_byte_size = size;
        P<T>::p_alloc(data, size);
    }

    virtual ~SharedBuffer() {
        P<T>::p_dealloc(m_allocated_buffer, m_byte_size);
        m_aligned_buffer = nullptr;
        m_allocated_buffer = nullptr;
        m_byte_size = 0;
    }

private:
    T _shared_object;
};

/// \brief SharedStreamBuffer class to store pointer to pre-acclocated buffer and provide streambuf interface.
///  Can return ptr to shared memory and its size
class SharedStreamBuffer : public std::streambuf {
public:
    SharedStreamBuffer(char* data, size_t size) : m_data(data), m_size(size), m_offset(0) {}

protected:
    // override std::streambuf methods
    std::streamsize xsgetn(char* s, std::streamsize count) override {
        auto real_count = std::min<std::streamsize>(m_size - m_offset, count);
        std::memcpy(s, m_data + m_offset, real_count);
        m_offset += real_count;
        return real_count;
    }

    int_type underflow() override {
        return (m_size == m_offset) ? traits_type::eof() : traits_type::to_int_type(*(m_data + m_offset));
    }

    int_type uflow() override {
        return (m_size == m_offset) ? traits_type::eof() : traits_type::to_int_type(*(m_data + m_offset++));
    }

    std::streamsize showmanyc() override {
        return m_size - m_offset;
    }

    pos_type seekoff(off_type off,
                     std::ios_base::seekdir dir,
                     std::ios_base::openmode which = std::ios_base::in) override {
        if (dir != std::ios_base::cur || which != std::ios_base::in) {
            return pos_type(off_type(-1));
        }
        m_offset += off;
        return pos_type(m_offset);
    }

    char* m_data;
    size_t m_size;
    size_t m_offset;
};

/// \brief OwningSharedStreamBuffer is a SharedStreamBuffer which owns its shared object.
class OwningSharedStreamBuffer : public SharedStreamBuffer {
public:
    OwningSharedStreamBuffer(std::shared_ptr<ov::AlignedBuffer> buffer)
        : SharedStreamBuffer(static_cast<char*>(buffer->get_ptr()), buffer->size()),
          m_shared_obj(buffer) {}

    std::shared_ptr<ov::AlignedBuffer> get_buffer() {
        return m_shared_obj;
    }

protected:
    std::shared_ptr<ov::AlignedBuffer> m_shared_obj;
};
}  // namespace ov

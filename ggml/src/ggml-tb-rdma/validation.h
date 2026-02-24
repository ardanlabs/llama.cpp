#pragma once

#include <cstddef>
#include <cstdint>

namespace ggml_tb_rdma {

// Returns true if `[offset, offset+size)` lies entirely within
// `[0, buffer_size)`. Detects integer overflow in `offset + size` — neither
// argument is implicitly trusted.
//
// `size == 0` is allowed (degenerate range).
inline bool validate_tensor_range(uint64_t buffer_size, uint64_t offset, uint64_t size) {
    if (size == 0) return offset <= buffer_size;
    if (offset > buffer_size) return false;
    const uint64_t remaining = buffer_size - offset;
    return size <= remaining;
}

// Multiplies `a * b` into `*out`, returning false on overflow. Used to
// bounds-check graph_compute payload sizes before allocation.
inline bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t * out) {
    if (a == 0 || b == 0) { *out = 0; return true; }
    if (a > UINT64_MAX / b) return false;
    *out = a * b;
    return true;
}

// `*out += addend`, false on overflow.
inline bool checked_add_u64(uint64_t addend, uint64_t * out) {
    if (addend > UINT64_MAX - *out) return false;
    *out += addend;
    return true;
}

} // namespace ggml_tb_rdma

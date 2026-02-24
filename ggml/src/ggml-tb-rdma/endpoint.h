#pragma once

#include <cstdint>
#include <string>

namespace ggml_tb_rdma {

// Result of parsing an endpoint string. `host` does not contain surrounding
// brackets for IPv6 literals. `port` is in [1, 65535].
struct Endpoint {
    std::string host;
    uint16_t    port = 0;
};

// Parse "host:port" or "[ipv6]:port".
//
// Returns true on success. On failure, leaves `out` untouched and (when non-
// null) writes a short, human-readable diagnostic to `*err`.
//
// Rules:
//   - Bare IPv6 literals MUST be bracketed: "[::1]:5000" parses; "::1:5000"
//     does not.
//   - Port: decimal 1..65535. No leading whitespace, no trailing garbage.
//   - Empty host is rejected.
bool parse_endpoint(const std::string & input, Endpoint & out, std::string * err = nullptr);

} // namespace ggml_tb_rdma

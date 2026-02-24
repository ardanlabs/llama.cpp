#include "endpoint.h"

#include <cerrno>
#include <cstdlib>

namespace ggml_tb_rdma {

static bool set_err(std::string * err, const char * msg) {
    if (err) *err = msg;
    return false;
}

bool parse_endpoint(const std::string & input, Endpoint & out, std::string * err) {
    if (input.empty()) return set_err(err, "empty endpoint");

    size_t host_begin = 0;
    size_t host_end   = std::string::npos;
    size_t port_pos   = std::string::npos;

    if (input.front() == '[') {
        // Bracketed IPv6 literal.
        size_t close = input.find(']');
        if (close == std::string::npos) return set_err(err, "missing ']' in IPv6 endpoint");
        if (close + 1 >= input.size() || input[close + 1] != ':') {
            return set_err(err, "expected ':<port>' after ']'");
        }
        host_begin = 1;
        host_end   = close;
        port_pos   = close + 2;
    } else {
        // Bare host:port. Require exactly one colon — rejects ambiguous
        // unbracketed IPv6 literals.
        size_t first = input.find(':');
        size_t last  = input.rfind(':');
        if (first == std::string::npos)        return set_err(err, "missing port");
        if (first != last)                     return set_err(err, "ambiguous endpoint — bracket IPv6 literals");
        host_begin = 0;
        host_end   = first;
        port_pos   = first + 1;
    }

    if (host_end <= host_begin) return set_err(err, "empty host");
    if (port_pos >= input.size()) return set_err(err, "empty port");

    std::string host = input.substr(host_begin, host_end - host_begin);
    std::string port = input.substr(port_pos);

    // Port: strict decimal 1..65535.
    if (port.empty()) return set_err(err, "empty port");
    for (char c : port) {
        if (c < '0' || c > '9') return set_err(err, "non-numeric port");
    }

    errno = 0;
    char * end = nullptr;
    unsigned long v = std::strtoul(port.c_str(), &end, 10);
    if (errno != 0 || end == nullptr || *end != '\0') return set_err(err, "invalid port");
    if (v < 1 || v > 65535)                            return set_err(err, "port out of range");

    out.host = std::move(host);
    out.port = static_cast<uint16_t>(v);
    return true;
}

} // namespace ggml_tb_rdma

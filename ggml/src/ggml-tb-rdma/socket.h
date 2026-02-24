#pragma once

// macOS-only socket helpers used by the Thunderbolt-RDMA TCP bootstrap. The
// TCP socket exists only long enough for one hello round-trip (after which
// M2 promotes the session to RDMA SEND/RECV) but it still has to be correct:
// EINTR retry, SO_NOSIGPIPE per-socket, getaddrinfo for IPv4+IPv6, and an
// explicit distinction between orderly close and error.

#include "endpoint.h"

#include <cstddef>
#include <string>

namespace ggml_tb_rdma {

// Result of a single send/recv operation.
enum class IoResult {
    Ok,        // requested byte count fully transferred
    Closed,    // peer performed orderly shutdown (recv only)
    Error,     // OS-level error; check `errno` for details
};

// Connect to host:port. Tries every address from getaddrinfo() until one
// succeeds. Returns -1 on failure (and writes a diagnostic to *err if non-
// null). The returned fd has TCP_NODELAY and SO_NOSIGPIPE set.
int tcp_connect(const Endpoint & ep, std::string * err = nullptr);

// Listen on host:port. Tries every getaddrinfo() entry; binds + listens on
// the first that works. Returns -1 on failure.
int tcp_listen(const Endpoint & ep, std::string * err = nullptr);

// Accept a connection from `listen_fd`. Applies SO_NOSIGPIPE + TCP_NODELAY
// on the returned fd. Returns -1 on failure.
int tcp_accept(int listen_fd, std::string * err = nullptr);

// Close a socket. Safe to pass -1. EINTR-tolerant.
void tcp_close(int fd);

// Send exactly `size` bytes (loops on partial writes; retries EINTR).
IoResult tcp_send_all(int fd, const void * data, size_t size);

// Receive exactly `size` bytes (loops on partial reads; retries EINTR).
// Returns `Closed` if the peer shuts down before `size` bytes arrive AND no
// bytes were received in the current call. A truncated transfer mid-stream
// returns `Error`.
IoResult tcp_recv_all(int fd, void * data, size_t size);

} // namespace ggml_tb_rdma

#include "socket.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>

namespace ggml_tb_rdma {

static void set_err(std::string * err, const char * prefix) {
    if (!err) return;
    int e = errno;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s: %s", prefix, std::strerror(e));
    *err = buf;
}

static bool apply_socket_options(int fd) {
    int one = 1;
    // SO_NOSIGPIPE is macOS-native; avoids the need for a process-global
    // signal(SIGPIPE, SIG_IGN).
    if (setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one)) != 0) {
        return false;
    }
    // Disable Nagle on the bootstrap socket — it carries exactly one hello
    // round-trip, so latency >> throughput.
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    return true;
}

int tcp_connect(const Endpoint & ep, std::string * err) {
    addrinfo hints = {};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    char port_buf[8];
    std::snprintf(port_buf, sizeof(port_buf), "%u", ep.port);

    addrinfo * res = nullptr;
    int rc = getaddrinfo(ep.host.c_str(), port_buf, &hints, &res);
    if (rc != 0) {
        if (err) *err = std::string("getaddrinfo: ") + gai_strerror(rc);
        return -1;
    }

    int fd = -1;
    int last_errno = 0;
    for (addrinfo * ai = res; ai != nullptr; ai = ai->ai_next) {
        fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) { last_errno = errno; continue; }

        if (!apply_socket_options(fd)) {
            last_errno = errno;
            ::close(fd);
            fd = -1;
            continue;
        }

        int cr;
        do {
            cr = ::connect(fd, ai->ai_addr, ai->ai_addrlen);
        } while (cr < 0 && errno == EINTR);

        if (cr == 0) break;
        last_errno = errno;
        ::close(fd);
        fd = -1;
    }

    freeaddrinfo(res);

    if (fd < 0) {
        errno = last_errno ? last_errno : ECONNREFUSED;
        set_err(err, "connect");
    }
    return fd;
}

int tcp_listen(const Endpoint & ep, std::string * err) {
    addrinfo hints = {};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags    = AI_PASSIVE;

    char port_buf[8];
    std::snprintf(port_buf, sizeof(port_buf), "%u", ep.port);

    addrinfo * res = nullptr;
    int rc = getaddrinfo(ep.host.c_str(), port_buf, &hints, &res);
    if (rc != 0) {
        if (err) *err = std::string("getaddrinfo: ") + gai_strerror(rc);
        return -1;
    }

    int fd = -1;
    int last_errno = 0;
    for (addrinfo * ai = res; ai != nullptr; ai = ai->ai_next) {
        fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) { last_errno = errno; continue; }

        if (!apply_socket_options(fd)) {
            last_errno = errno;
            ::close(fd);
            fd = -1;
            continue;
        }

        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

        if (::bind(fd, ai->ai_addr, ai->ai_addrlen) != 0) {
            last_errno = errno;
            ::close(fd);
            fd = -1;
            continue;
        }
        if (::listen(fd, 1) != 0) {
            last_errno = errno;
            ::close(fd);
            fd = -1;
            continue;
        }
        break;
    }

    freeaddrinfo(res);

    if (fd < 0) {
        errno = last_errno ? last_errno : EADDRNOTAVAIL;
        set_err(err, "listen");
    }
    return fd;
}

int tcp_accept(int listen_fd, std::string * err) {
    int fd;
    do {
        fd = ::accept(listen_fd, nullptr, nullptr);
    } while (fd < 0 && errno == EINTR);

    if (fd < 0) {
        set_err(err, "accept");
        return -1;
    }
    if (!apply_socket_options(fd)) {
        set_err(err, "setsockopt");
        ::close(fd);
        return -1;
    }
    return fd;
}

void tcp_close(int fd) {
    if (fd < 0) return;
    int rc;
    do {
        rc = ::close(fd);
    } while (rc < 0 && errno == EINTR);
}

IoResult tcp_send_all(int fd, const void * data, size_t size) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    size_t sent = 0;
    while (sent < size) {
        ssize_t n = ::send(fd, p + sent, size - sent, 0);
        if (n > 0) {
            sent += static_cast<size_t>(n);
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        return IoResult::Error;
    }
    return IoResult::Ok;
}

IoResult tcp_recv_all(int fd, void * data, size_t size) {
    uint8_t * p = static_cast<uint8_t *>(data);
    size_t got = 0;
    while (got < size) {
        ssize_t n = ::recv(fd, p + got, size - got, 0);
        if (n > 0) {
            got += static_cast<size_t>(n);
            continue;
        }
        if (n == 0) {
            // Orderly shutdown. If we already got some bytes, that's a
            // truncated transfer — report Error so callers can't treat
            // partially-received frames as valid.
            return got == 0 ? IoResult::Closed : IoResult::Error;
        }
        if (errno == EINTR) continue;
        return IoResult::Error;
    }
    return IoResult::Ok;
}

} // namespace ggml_tb_rdma

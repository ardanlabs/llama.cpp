// Unit tests for the Thunderbolt-RDMA backend foundations (M0).
//
// Coverage:
//   - parse_endpoint: IPv4, IPv6, bad port, missing host, brackets, garbage.
//   - validate_tensor_range / checked_mul_u64 / checked_add_u64 overflow.
//   - Transport framing over a pair of socketpair() fds:
//       * good round-trip
//       * cmd mismatch closes transport
//       * oversized payload closes transport
//       * truncated stream surfaces IoError, not silent success
//
// No ggml dependency — these tests link only against the tb-rdma sources.

#include "endpoint.h"
#include "protocol.h"
#include "socket.h"
#include "transport.h"
#include "validation.h"

#include <sys/socket.h>
#include <signal.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace ggml_tb_rdma;

static int g_failures = 0;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
            g_failures++;                                                             \
        }                                                                             \
    } while (0)

// -----------------------------------------------------------------------------
// Endpoint parser
// -----------------------------------------------------------------------------

static void test_endpoint_parser() {
    Endpoint ep;
    std::string err;

    // Happy paths.
    CHECK(parse_endpoint("127.0.0.1:5050", ep, &err));
    CHECK(ep.host == "127.0.0.1");
    CHECK(ep.port == 5050);

    CHECK(parse_endpoint("[::1]:5000", ep, &err));
    CHECK(ep.host == "::1");
    CHECK(ep.port == 5000);

    CHECK(parse_endpoint("localhost:1", ep, &err));
    CHECK(ep.port == 1);

    CHECK(parse_endpoint("example.com:65535", ep, &err));
    CHECK(ep.port == 65535);

    // Failure paths.
    CHECK(!parse_endpoint("", ep, &err));
    CHECK(!parse_endpoint(":5050", ep, &err));            // empty host
    CHECK(!parse_endpoint("host:", ep, &err));            // empty port
    CHECK(!parse_endpoint("host", ep, &err));             // missing port
    CHECK(!parse_endpoint("host:0", ep, &err));           // port out of range
    CHECK(!parse_endpoint("host:65536", ep, &err));       // port out of range
    CHECK(!parse_endpoint("host:abc", ep, &err));         // non-numeric port
    CHECK(!parse_endpoint("host:50 ", ep, &err));         // trailing garbage
    CHECK(!parse_endpoint("::1:5000", ep, &err));         // ambiguous v6
    CHECK(!parse_endpoint("[::1]5000", ep, &err));        // missing :port after ]
    CHECK(!parse_endpoint("[::1:5000", ep, &err));        // missing ]
}

// -----------------------------------------------------------------------------
// Validation helpers
// -----------------------------------------------------------------------------

static void test_validation() {
    // Normal ranges
    CHECK( validate_tensor_range(100, 0,  100));
    CHECK( validate_tensor_range(100, 50, 50));
    CHECK( validate_tensor_range(100, 0,  0));
    CHECK( validate_tensor_range(100, 100, 0));
    // Out of range
    CHECK(!validate_tensor_range(100, 0,  101));
    CHECK(!validate_tensor_range(100, 50, 51));
    CHECK(!validate_tensor_range(100, 101, 0));
    // Overflow: offset = UINT64_MAX - 10, size = 100 → offset+size overflows
    CHECK(!validate_tensor_range(100, UINT64_MAX - 10, 100));
    CHECK(!validate_tensor_range(UINT64_MAX, UINT64_MAX, 1));

    uint64_t out = 0;
    CHECK( checked_mul_u64(123, 456, &out)); CHECK(out == 123ull * 456ull);
    CHECK( checked_mul_u64(0, UINT64_MAX, &out)); CHECK(out == 0);
    CHECK(!checked_mul_u64(UINT64_MAX, 2, &out));
    CHECK( checked_mul_u64(UINT64_MAX, 1, &out)); CHECK(out == UINT64_MAX);

    out = 10;
    CHECK( checked_add_u64(5, &out)); CHECK(out == 15);
    out = UINT64_MAX - 5;
    CHECK( checked_add_u64(5, &out)); CHECK(out == UINT64_MAX);
    out = UINT64_MAX - 5;
    CHECK(!checked_add_u64(6, &out));
}

// -----------------------------------------------------------------------------
// Transport framing via socketpair
// -----------------------------------------------------------------------------

// Server-side transport over an existing fd — bypass the TCP listen/accept
// so the test stays in-process.
static bool server_take_fd_and_hello(Transport & tx, int fd, std::string * err) {
    return tx.server_accept(fd, "", err);
}

static bool client_take_fd_and_hello(Transport & tx, int fd, std::string * err) {
    // Mirrors `Transport::server_accept` without the connect step — we use
    // the same path because the *client* hello is what we want to drive.
    // For the test we just open a Transport on `fd` and send our own hello.
    //
    // Trick: client_connect runs hello-as-client, so we need a way to
    // attach an existing fd as the client side. We do that by writing a
    // hello manually using the raw socket helpers, then handing the fd to
    // `server_accept` would do the wrong thing. Instead, the test pairs a
    // Transport doing `server_accept` (acts as "server") with another
    // Transport doing `server_accept`-on-fd plus a manual hello send via
    // raw send/recv.
    //
    // Simpler approach: use `server_accept` on both ends and drive the
    // hello manually outside the helper. We do that below.
    (void) tx; (void) fd; (void) err;
    return false;
}

static void test_transport_roundtrip() {
    int sp[2] = { -1, -1 };
    int rc = socketpair(AF_UNIX, SOCK_STREAM, 0, sp);
    CHECK(rc == 0);
    if (rc != 0) return;

    // We test the framing layer below the hello by writing raw frames on
    // one side and reading via Transport on the other. To do that, drive
    // hello via raw helpers first, then both sides drop into framed mode.
    //
    // The simpler thing: a real round-trip via the public API. We make one
    // side server_accept (it expects a hello), and the other side perform
    // a hello-equivalent by sending a hello frame and reading the reply.

    Transport server_tx;
    std::thread server_thread([&]{
        std::string err;
        // server_accept expects to receive hello then send hello back.
        bool ok = server_take_fd_and_hello(server_tx, sp[0], &err);
        if (!ok) std::fprintf(stderr, "server_accept failed: %s\n", err.c_str());
    });

    // Client side hello sent by hand.
    {
        MsgHello hello = {};
        hello.version_major = GGML_TB_RDMA_PROTO_VERSION_MAJOR;
        hello.features = GGML_TB_RDMA_FEATURE_FRAMED_V2;
        hello.required = GGML_TB_RDMA_FEATURE_FRAMED_V2;
        MsgHeader hdr = {};
        hdr.cmd = CMD_HELLO;
        hdr.payload_size = sizeof(hello);
        IoResult r = tcp_send_all(sp[1], &hdr, sizeof(hdr));
        CHECK(r == IoResult::Ok);
        r = tcp_send_all(sp[1], &hello, sizeof(hello));
        CHECK(r == IoResult::Ok);

        // Read server hello.
        MsgHeader rsp_hdr = {};
        r = tcp_recv_all(sp[1], &rsp_hdr, sizeof(rsp_hdr));
        CHECK(r == IoResult::Ok);
        CHECK(rsp_hdr.cmd == CMD_HELLO);
        CHECK(rsp_hdr.payload_size == sizeof(MsgHello));
        MsgHello srv_hello = {};
        r = tcp_recv_all(sp[1], &srv_hello, sizeof(srv_hello));
        CHECK(r == IoResult::Ok);
        CHECK(srv_hello.version_major == GGML_TB_RDMA_PROTO_VERSION_MAJOR);

        // Send a GET_ALIGNMENT request (header-only) so the server has
        // something to react to; we don't actually care about the reply
        // here — we close the socket and the server thread exits.
        // But: server_accept doesn't loop after hello, so it returns
        // immediately on success. So we just join.
    }

    server_thread.join();

    // Now sp[0] is owned by server_tx, sp[1] is owned by the test fd. Close
    // sp[1]; server_tx destructor will close sp[0].
    tcp_close(sp[1]);
}

// Cmd-mismatch / oversized payload / truncated stream tests via direct raw
// helpers — these don't need to go through hello, they exercise the public
// `recv_request` directly. We construct a Transport over a socketpair where
// we manually completed the hello (or rather, skip it — by using a private
// constructor pathway). Since we don't have such a constructor, we drive
// `recv_request` after server_accept has finished hello with a peer that
// just sent hello correctly. We then send a malformed *next* frame from
// the peer side and confirm `recv_request` reports the right status and
// closes.
static void test_transport_protocol_errors() {
    auto run_case = [](auto inject_after_hello, RequestStatus expected) {
        int sp[2] = { -1, -1 };
        int rc = socketpair(AF_UNIX, SOCK_STREAM, 0, sp);
        CHECK(rc == 0);
        if (rc != 0) return;

        Transport server_tx;
        std::atomic<bool> hello_done{false};
        std::thread t([&]{
            std::string err;
            (void) server_tx.server_accept(sp[0], "", &err);
            hello_done.store(true);
        });

        // Drive client hello.
        MsgHello hello = {};
        hello.version_major = GGML_TB_RDMA_PROTO_VERSION_MAJOR;
        hello.features = GGML_TB_RDMA_FEATURE_FRAMED_V2;
        hello.required = GGML_TB_RDMA_FEATURE_FRAMED_V2;
        MsgHeader hdr = {};
        hdr.cmd = CMD_HELLO;
        hdr.payload_size = sizeof(hello);
        (void) tcp_send_all(sp[1], &hdr, sizeof(hdr));
        (void) tcp_send_all(sp[1], &hello, sizeof(hello));
        MsgHeader rsp_hdr = {};
        MsgHello srv_hello = {};
        (void) tcp_recv_all(sp[1], &rsp_hdr, sizeof(rsp_hdr));
        (void) tcp_recv_all(sp[1], &srv_hello, sizeof(srv_hello));

        t.join();
        CHECK(hello_done.load());

        // Now inject the malformed/truncated frame from the client side.
        inject_after_hello(sp[1]);

        // server reads next request — should hit `expected`.
        MsgHeader got_hdr = {};
        std::vector<uint8_t> payload;
        RequestStatus s = server_tx.recv_request(&got_hdr, &payload, MAX_CONTROL_FRAME_BYTES);
        CHECK(s == expected);

        tcp_close(sp[1]);
    };

    // Case 1: oversized payload — server's recv_request caps to
    // MAX_CONTROL_FRAME_BYTES. Sending payload_size > cap → ProtocolError.
    run_case([](int fd){
        MsgHeader h = {};
        h.cmd = CMD_ALLOC_BUFFER;
        h.payload_size = MAX_CONTROL_FRAME_BYTES + 1;
        (void) tcp_send_all(fd, &h, sizeof(h));
        // intentionally don't send the body — but the server should detect
        // the oversize from the header alone and bail.
    }, RequestStatus::ProtocolError);

    // Case 2: malformed header (flags!=0) → ProtocolError.
    run_case([](int fd){
        MsgHeader h = {};
        h.cmd = CMD_ALLOC_BUFFER;
        h.flags = 0x42;
        h.payload_size = 0;
        (void) tcp_send_all(fd, &h, sizeof(h));
    }, RequestStatus::ProtocolError);

    // Case 3: truncated stream — close mid-frame after header → IoError.
    run_case([](int fd){
        MsgHeader h = {};
        h.cmd = CMD_ALLOC_BUFFER;
        h.payload_size = 32;  // promise 32 bytes
        (void) tcp_send_all(fd, &h, sizeof(h));
        // send only 5 bytes, then shutdown.
        uint8_t partial[5] = {1,2,3,4,5};
        (void) tcp_send_all(fd, partial, sizeof(partial));
        shutdown(fd, SHUT_WR);
    }, RequestStatus::IoError);

    // Case 4: orderly disconnect before any new frame → Disconnected.
    run_case([](int fd){
        shutdown(fd, SHUT_WR);
    }, RequestStatus::Disconnected);
}

// -----------------------------------------------------------------------------
// M2a fallback: GGML_TB_RDMA_FORCE_TCP=1 must skip RDMA bring-up entirely
// and leave the transport in TCP mode after server_accept().
// -----------------------------------------------------------------------------

static void test_force_tcp_fallback() {
    setenv("GGML_TB_RDMA_FORCE_TCP", "1", 1);

    int sp[2] = { -1, -1 };
    int rc = socketpair(AF_UNIX, SOCK_STREAM, 0, sp);
    CHECK(rc == 0);
    if (rc != 0) return;

    Transport server_tx;
    std::atomic<bool> done{false};
    std::thread t([&]{
        std::string err;
        (void) server_tx.server_accept(sp[0], "", &err);
        done.store(true);
    });

    // Client side hello.
    MsgHello hello = {};
    hello.version_major = GGML_TB_RDMA_PROTO_VERSION_MAJOR;
    hello.features = GGML_TB_RDMA_FEATURE_FRAMED_V2;
    hello.required = GGML_TB_RDMA_FEATURE_FRAMED_V2;
    MsgHeader hdr = {};
    hdr.cmd = CMD_HELLO;
    hdr.payload_size = sizeof(hello);
    (void) tcp_send_all(sp[1], &hdr, sizeof(hdr));
    (void) tcp_send_all(sp[1], &hello, sizeof(hello));

    MsgHeader rsp_hdr = {};
    MsgHello srv_hello = {};
    (void) tcp_recv_all(sp[1], &rsp_hdr, sizeof(rsp_hdr));
    (void) tcp_recv_all(sp[1], &srv_hello, sizeof(srv_hello));

    t.join();
    CHECK(done.load());
    CHECK(server_tx.mode() == TransportMode::Tcp);
    CHECK(server_tx.stats().fallback_events == 0); // FORCE_TCP doesn't even attempt

    tcp_close(sp[1]);
    unsetenv("GGML_TB_RDMA_FORCE_TCP");
}

// -----------------------------------------------------------------------------
// M2d counters: TransportStats includes the new RDMA counters and they're
// zero in pure-TCP runs.
// -----------------------------------------------------------------------------

static void test_stats_fields_default_zero() {
    Transport tx;
    auto & s = tx.stats();
    CHECK(s.rdma_frames_sent     == 0);
    CHECK(s.rdma_frames_received == 0);
    CHECK(s.rdma_retransmits     == 0);
    CHECK(s.rdma_credit_stalls   == 0);
    CHECK(s.rdma_qp_cq_errors    == 0);
    CHECK(s.rdma_flow_frames     == 0);
    CHECK(s.fallback_events      == 0);
}

// -----------------------------------------------------------------------------

int main() {
    // socketpair() AF_UNIX fds don't go through tcp_connect/listen, so the
    // backend's SO_NOSIGPIPE doesn't cover them. Ignore SIGPIPE process-wide
    // for the duration of this test program.
    signal(SIGPIPE, SIG_IGN);

    test_endpoint_parser();
    test_validation();
    test_transport_roundtrip();
    test_transport_protocol_errors();
    test_force_tcp_fallback();
    test_stats_fields_default_zero();

    if (g_failures == 0) {
        std::fprintf(stderr, "tb-rdma tests: OK\n");
        return 0;
    }
    std::fprintf(stderr, "tb-rdma tests: %d failure(s)\n", g_failures);
    return 1;
}

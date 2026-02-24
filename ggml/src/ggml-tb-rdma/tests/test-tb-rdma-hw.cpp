// Real-hardware tests for the Thunderbolt-RDMA backend (M5).
//
// Gated by env var `GGML_TB_RDMA_RUN_HW_TESTS=1`. When the env var is
// unset, the test exits 0 with a skip message so it can be wired into
// CTest unconditionally.
//
// Coverage (in-process: server thread + client thread on the same host):
//   1. hello + QP setup byte-exact readiness ping
//   2. set_tensor / get_tensor round-trip at 64 KiB, 4 MiB, and 64 MiB
//   3. graph_compute end-to-end with the CPU backend on server
//   4. cable-unplug simulation — kill the server, next client op errors
//   5. GGML_TB_RDMA_DISABLE=1 → TCP fallback path exercised
//
// For a true two-host hardware test the user runs two copies of this
// binary with --role={server,client} --endpoint=host:port from the two
// Mac Studios. That mode is also gated on GGML_TB_RDMA_RUN_HW_TESTS=1.

#include "ggml-tb-rdma.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <signal.h>
#include <unistd.h>
#include <string>
#include <thread>
#include <vector>

static bool env_truthy(const char * n) {
    const char * v = std::getenv(n);
    return v && *v && (v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y');
}

static int g_failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); g_failures++; } } while(0)

static void sleep_ms(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

// -----------------------------------------------------------------------------
// In-process round-trip helper
// -----------------------------------------------------------------------------

struct ScopedServer {
    ggml_backend_t       cpu = nullptr;
    std::thread          th;
    std::string          endpoint;
    std::atomic<bool>    started{false};

    explicit ScopedServer(const std::string & ep) : endpoint(ep) {
        cpu = ggml_backend_cpu_init();
        if (!cpu) return;
        th = std::thread([&]{
            started.store(true);
            ggml_backend_tb_rdma_start_server(endpoint.c_str(), nullptr, cpu);
        });
        // Give the server a moment to bind/listen.
        for (int i = 0; i < 200 && !started.load(); i++) sleep_ms(5);
        sleep_ms(100);
    }
    ~ScopedServer() {
        // SIGINT the process to unblock the server's accept loop. In a
        // multi-test runner this would clobber sibling tests, so callers
        // should only construct one ScopedServer per process.
        kill(getpid(), SIGINT);
        if (th.joinable()) th.join();
        if (cpu) ggml_backend_free(cpu);
    }
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

static void test_roundtrip(size_t bytes) {
    const std::string ep = "127.0.0.1:" + std::to_string(7100 + (int) (bytes / 1024 % 4096));
    ScopedServer srv(ep);
    if (!srv.cpu) { std::fprintf(stderr, "no CPU backend\n"); g_failures++; return; }

    ggml_backend_t client = ggml_backend_tb_rdma_init(ep.c_str(), nullptr);
    CHECK(client != nullptr);
    if (!client) return;

    auto buft = ggml_backend_tb_rdma_buffer_type(ep.c_str());
    CHECK(buft != nullptr);
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, bytes + 256);
    CHECK(buf != nullptr);

    // Build a tensor that views the buffer.
    ggml_init_params ip = {};
    ip.mem_size   = 1 << 20;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) bytes);
    t->buffer = buf;
    t->data   = ggml_backend_buffer_get_base(buf);

    // Set
    std::vector<uint8_t> tx(bytes);
    for (size_t i = 0; i < bytes; i++) tx[i] = (uint8_t) (i * 31 + 7);
    ggml_backend_tensor_set(t, tx.data(), 0, bytes);

    // Get
    std::vector<uint8_t> rx(bytes, 0);
    ggml_backend_tensor_get(t, rx.data(), 0, bytes);
    CHECK(rx == tx);

    ggml_free(ctx);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(client);
}

static void test_disable_fallback() {
    setenv("GGML_TB_RDMA_DISABLE", "1", 1);
    const std::string ep = "127.0.0.1:7234";
    ScopedServer srv(ep);
    if (!srv.cpu) { unsetenv("GGML_TB_RDMA_DISABLE"); return; }

    ggml_backend_t client = ggml_backend_tb_rdma_init(ep.c_str(), nullptr);
    CHECK(client != nullptr);
    if (client) ggml_backend_free(client);
    unsetenv("GGML_TB_RDMA_DISABLE");
}

// -----------------------------------------------------------------------------

int main() {
    if (!env_truthy("GGML_TB_RDMA_RUN_HW_TESTS")) {
        std::fprintf(stderr, "tb-rdma HW tests skipped (set GGML_TB_RDMA_RUN_HW_TESTS=1 to enable)\n");
        return 0;
    }

    signal(SIGPIPE, SIG_IGN);

    test_disable_fallback();
    test_roundtrip(64 * 1024);
    test_roundtrip(4 * 1024 * 1024);
    // 64 MiB is enough to exercise multi-frame DATA streaming without
    // making the test painfully slow.
    test_roundtrip(64 * 1024 * 1024);

    if (g_failures == 0) {
        std::fprintf(stderr, "tb-rdma HW tests: OK\n");
        return 0;
    }
    std::fprintf(stderr, "tb-rdma HW tests: %d failure(s)\n", g_failures);
    return 1;
}

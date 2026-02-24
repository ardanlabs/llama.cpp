# TB-RDMA backend (Thunderbolt RDMA, macOS)

`ggml-tb-rdma` is a macOS-only backend that talks to a remote `ggml`
backend. It bootstraps over TCP for one round-trip (hello + QP info) and
then runs all command + data traffic over Apple Thunderbolt UC RDMA
SEND/RECV. If RDMA bring-up fails at connection time, the session
transparently falls back to running everything over TCP.

It is not built on Linux or Windows. For RDMA over RoCE/IB on Linux, use
the regular `ggml-rpc` backend with `-DGGML_RPC_RDMA=ON` instead. The
two features share no code.

## Build

```
cmake -B build -DGGML_TB_RDMA=ON
cmake --build build --target ggml-tb-rdma
```

The option is `ON` by default on Apple and force-`OFF` elsewhere. The
backend is built as a shared library `libggml-tb-rdma.dylib` and loads
`librdma.dylib` lazily at runtime via `dlopen()` — there is no link-time
dependency, so the binary still runs on Macs without Apple's RDMA stack
enabled (it will simply fall back to TCP).

## Status (M0 → M2 + M5)

Implemented:

* Strict frame validation, fail-closed on protocol error.
* Hello negotiation with major-version and feature-bit checks.
* Hardened TCP bootstrap (getaddrinfo for v4/v6, bracket-aware endpoint
  parser, `EINTR` retry, per-socket `SO_NOSIGPIPE`).
* Per-connection transport mutex (one in-flight op).
* Full client/server handlers for `alloc_buffer`, `free_buffer`,
  `get_base`, `clear`, `memset`, `set_tensor`, `get_tensor`,
  `copy_tensor`, `init_tensor`, `graph_compute`, `get_device_memory`,
  `get_alignment`, `get_max_size`.
* RDMA bring-up: device + GID selection (auto-match against the local
  bootstrap-TCP IP, with `GGML_TB_RDMA_GID_INDEX` override), UC QP
  creation, INIT → RTR → RTS transitions, slot pools sized from queried
  device caps, post-handshake SEND/RECV readiness ping.
* Reliable CONTROL stream over RDMA SEND/RECV: per-frame go-back-N with
  cumulative ACK, ACK-only FLOW frames on idle / stall, single-threaded
  synchronous progress engine, hard timeout after `MAX_RETRANSMITS` (3
  by default).
* DATA path: every chunk is one path-MTU-sized SEND; on the receive
  side, frames are copied straight into the destination buffer with no
  full-message reassembly buffer (a 1 MiB cap covers only CONTROL).
* Connection-time TCP fallback with a clear log line; `GGML_TB_RDMA_-
  REQUIRE_RDMA=1` upgrades the fallback to a hard error.
* Counters: bytes / frames sent + received, retransmits, credit stalls,
  QP/CQ errors, fallback events. All exposed via `TransportStats`.
* Graceful shutdown: SIGINT/SIGTERM closes the listen socket cleanly.

Real-hardware tests live in `tests/test-tb-rdma-hw.cpp` and are gated by
`GGML_TB_RDMA_RUN_HW_TESTS=1` (skipped otherwise so CI on a host without
librdma still passes).

## API

```c
#include "ggml-tb-rdma.h"

// Client
ggml_backend_t backend = ggml_backend_tb_rdma_init("192.168.1.10:5050", NULL);
ggml_backend_buffer_type_t buft = ggml_backend_tb_rdma_buffer_type("192.168.1.10:5050");

// Server
ggml_backend_tb_rdma_start_server("0.0.0.0:5050", NULL, local_backend);

// Device enumeration (lazily loads librdma.dylib)
bool         ok    = ggml_tb_rdma_available();
size_t       n     = ggml_tb_rdma_get_device_count();
const char * name0 = ggml_tb_rdma_get_device_name(0);
```

Endpoint strings:

* IPv4: `host:port` or `1.2.3.4:5050`
* IPv6: `[::1]:5050` (brackets required to disambiguate from the
  embedded `:` separators)

## Environment variables

| Name                           | Default | Effect                                                            |
|--------------------------------|---------|-------------------------------------------------------------------|
| `GGML_TB_RDMA_DEVICE`          | —       | RDMA device name to open (default: first available).              |
| `GGML_TB_RDMA_GID_INDEX`       | auto    | Port-1 GID table index to use; auto-matches the local TCP IP.     |
| `GGML_TB_RDMA_FORCE_TCP`       | unset   | Skip RDMA bring-up entirely, run the whole session over TCP.      |
| `GGML_TB_RDMA_DISABLE`         | unset   | Synonym for `FORCE_TCP`, accepted for symmetry with the M5 plan.  |
| `GGML_TB_RDMA_REQUIRE_RDMA`    | unset   | If RDMA bring-up fails, error out instead of falling back to TCP. |
| `GGML_TB_RDMA_PING_TIMEOUT_MS` | 5000    | Readiness-ping timeout (ms).                                      |
| `GGML_TB_RDMA_LOG_LEVEL`       | `error` | `silent` / `error` / `info` / `debug`.                            |
| `GGML_TB_RDMA_RUN_HW_TESTS`    | unset   | Enable the real-hardware test suite.                              |

## Apple Thunderbolt verbs constraints

This backend respects the limits documented in Apple TN3205:

* SEND/RECV only — no RDMA WRITE/READ.
* UC QPs only — silent drops possible; we layer go-back-N reliability
  on top.
* Up to 10 UC QPs per process. The single-client server model uses one
  QP per connection.
* No hardware remote-write or atomics — every byte arrives as an
  app-level SEND.
* MRs are registered against `posix_memalign`-allocated buffers, sized
  to a multiple of the system page size, because each Thunderbolt
  controller sits behind an IOMMU that only maps page-aligned regions.
  Memory is registered with `IBV_ACCESS_LOCAL_WRITE` only.
* The QP-info handshake carries the peer port LID from
  `ibv_query_port()`; the RTR transition uses `ah_attr.dlid = peer.lid`
  and `grh.hop_limit = 1`, matching the TN3205 example.
* Sender credit budget is initialized from the peer's advertised
  `recv_credits = R` and replenished one-for-one as the peer's
  cumulative ACK advances, capped at `R`.

See [`fix-rdma-plan.md`](../../fix-rdma-plan.md) for the full milestone
plan and rationale.

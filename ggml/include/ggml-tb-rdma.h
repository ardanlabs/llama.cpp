#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// Thunderbolt-RDMA backend (Apple-only)
//
// This backend targets macOS hosts connected via Thunderbolt 5, using Apple's
// Thunderbolt RDMA stack (verbs surface backed by `librdma.dylib`). It is NOT
// portable; the source tree only builds it on Darwin.
//
// Wire protocol negotiation uses feature bits — see the matching internal
// header for the full list. Major-version mismatch is fatal at hello time.
// -----------------------------------------------------------------------------

// Wire protocol major version. Incompatible-major mismatch closes the
// connection at hello. Bumped to 2 because the v1 wire (from the original
// `ggml-rpc_rdma` prototype) is not compatible with this implementation.
#define GGML_TB_RDMA_PROTO_VERSION_MAJOR 2

// Optional feature bits advertised in the hello message (uint32).
// Required bits missing on the peer → TCP fallback (or hard error if the
// caller sets GGML_TB_RDMA_REQUIRE_RDMA=1).
#define GGML_TB_RDMA_FEATURE_FRAMED_V2  0x00000001u
#define GGML_TB_RDMA_FEATURE_SEND_RECV  0x00000002u

// -----------------------------------------------------------------------------
// Client backend
// -----------------------------------------------------------------------------

// Initialize a client backend connected to the given remote endpoint.
//   endpoint    : "host:port" or "[ipv6]:port"
//   rdma_device : RDMA device name (e.g. "rdma_en2") or NULL for auto-detect
//                 (ignored if RDMA bring-up fails and the connection falls
//                 back to TCP at startup)
// Returns nullptr on failure.
GGML_BACKEND_API ggml_backend_t ggml_backend_tb_rdma_init(
    const char * endpoint,
    const char * rdma_device);

// Returns true if `backend` was created by ggml_backend_tb_rdma_init.
GGML_BACKEND_API bool ggml_backend_is_tb_rdma(ggml_backend_t backend);

// Buffer type backing remote allocations for the given endpoint. Resolves to
// the shared per-endpoint backend context (see `ggml_backend_tb_rdma_init`),
// so calling this before `_init` for the same endpoint returns a placeholder
// that becomes usable once `_init` runs.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_tb_rdma_buffer_type(
    const char * endpoint);

// Query the remote device's free/total memory (best-effort).
GGML_BACKEND_API void ggml_backend_tb_rdma_get_device_memory(
    const char * endpoint,
    size_t * free,
    size_t * total);

// -----------------------------------------------------------------------------
// Server
// -----------------------------------------------------------------------------

// Run the server on `endpoint` until SIGINT/SIGTERM or an unrecoverable error.
// This is a single-client server — a second incoming connection is rejected
// until the first closes.
//   endpoint    : "host:port" or "[ipv6]:port" to listen on
//   rdma_device : RDMA device or NULL for auto
//   backend     : local compute backend used to satisfy graph_compute calls
GGML_BACKEND_API void ggml_backend_tb_rdma_start_server(
    const char * endpoint,
    const char * rdma_device,
    ggml_backend_t backend);

// -----------------------------------------------------------------------------
// Device enumeration
// -----------------------------------------------------------------------------

GGML_BACKEND_API bool         ggml_tb_rdma_available(void);
GGML_BACKEND_API size_t       ggml_tb_rdma_get_device_count(void);
GGML_BACKEND_API const char * ggml_tb_rdma_get_device_name(size_t index);

// -----------------------------------------------------------------------------
// Backend registry
// -----------------------------------------------------------------------------

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_tb_rdma_reg(void);

#ifdef __cplusplus
}
#endif

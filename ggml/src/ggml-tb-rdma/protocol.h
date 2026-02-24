#pragma once

// -----------------------------------------------------------------------------
// Thunderbolt-RDMA wire protocol.
//
// All structures are packed and little-endian. Cross-endian communication is
// not supported. Major-version mismatch at hello is fatal.
// -----------------------------------------------------------------------------

#include "ggml-tb-rdma.h"

#include <cstdint>
#include <cstddef>

#pragma pack(push, 1)

namespace ggml_tb_rdma {

// -----------------------------------------------------------------------------
// Limits — kept conservative; tightened individually per command in handlers.
// -----------------------------------------------------------------------------

// Maximum byte-size of a single CONTROL frame payload (anything except bulk
// tensor data). This caps `MsgHeader.payload_size` for control commands.
//   - alloc/free/get_base/clear/memset replies are tiny.
//   - graph_compute has its own dedicated cap (see MAX_GRAPH_FRAME_BYTES).
//   - set/get_tensor (TCP path) data is *separately* bounded below.
constexpr uint32_t MAX_CONTROL_FRAME_BYTES = 1u << 20;  // 1 MiB

// Maximum byte-size of a CMD_GRAPH_COMPUTE request payload. Includes the
// fixed header struct, the node-id array, and the WireTensor array.
constexpr uint32_t MAX_GRAPH_FRAME_BYTES   = 64u << 20; // 64 MiB

// Maximum number of nodes / tensors in a CMD_GRAPH_COMPUTE request. The byte
// cap above is the primary guard; this is a sanity bound enforced first.
constexpr uint32_t MAX_GRAPH_NODES   = 1u << 20;
constexpr uint32_t MAX_GRAPH_TENSORS = 1u << 20;

// Maximum byte-size of a single CMD_SET_TENSOR or CMD_GET_TENSOR payload over
// TCP (data part). 256 MiB; large transfers belong on the RDMA SEND/RECV path
// once M2 lands.
constexpr uint64_t MAX_TENSOR_BYTES = 256ull << 20;

// Tensor serialization (matches ggml-rpc's WireTensor layout so the same
// helpers can be ported once M1.1 lands).
constexpr int MAX_DIMS       = 4;
constexpr int MAX_SRC        = 10;
constexpr int MAX_OP_PARAMS  = 64;
constexpr int MAX_NAME       = 64;

// -----------------------------------------------------------------------------
// Commands
// -----------------------------------------------------------------------------

enum TbRdmaCmd : uint8_t {
    // Connection management
    CMD_HELLO            = 0,
    CMD_GOODBYE          = 1,
    CMD_ERROR            = 2,  // payload: MsgError + optional detail bytes
    CMD_QP_INFO          = 3,  // M2a: post-hello QP destination exchange (TCP)
    CMD_RDMA_READY       = 4,  // M2a: readiness ping payload echoed over RDMA

    // Buffer management
    CMD_ALLOC_BUFFER     = 10,
    CMD_FREE_BUFFER      = 11,
    CMD_BUFFER_GET_BASE  = 12,
    CMD_BUFFER_CLEAR     = 13,
    CMD_BUFFER_MEMSET    = 14,  // new — wires through to remote (M1.4)

    // Tensor operations
    CMD_SET_TENSOR       = 20,
    CMD_GET_TENSOR       = 21,
    CMD_COPY_TENSOR      = 22,
    CMD_INIT_TENSOR      = 23,

    // Graph execution
    CMD_GRAPH_COMPUTE    = 30,

    // Memory / device info
    CMD_GET_DEVICE_MEMORY = 40,
    CMD_GET_ALIGNMENT     = 41,
    CMD_GET_MAX_SIZE      = 42,

    CMD_COUNT
};

// -----------------------------------------------------------------------------
// Common header — sent in front of every frame on the TCP control channel.
// -----------------------------------------------------------------------------

struct MsgHeader {
    uint8_t  cmd;
    uint8_t  flags;          // reserved, must be zero
    uint16_t reserved;       // must be zero
    uint32_t payload_size;   // bytes following the header for *this* frame
};
static_assert(sizeof(MsgHeader) == 8, "MsgHeader must be 8 bytes");

// -----------------------------------------------------------------------------
// Error payload (CMD_ERROR). detail_len bytes of UTF-8 follow.
// -----------------------------------------------------------------------------

enum TbRdmaError : uint16_t {
    ERR_NONE              = 0,
    ERR_PROTOCOL          = 1,  // framing / cmd mismatch / bad payload size
    ERR_UNSUPPORTED       = 2,
    ERR_OOM               = 3,
    ERR_INVALID_ARGUMENT  = 4,
    ERR_BAD_BUFFER        = 5,
    ERR_BAD_TENSOR        = 6,
    ERR_NOT_AVAILABLE     = 7,
    ERR_INTERNAL          = 8,
};

struct MsgError {
    uint16_t code;          // TbRdmaError
    uint16_t reserved;
    uint32_t detail_len;    // bytes of UTF-8 detail message following
};
static_assert(sizeof(MsgError) == 8, "MsgError must be 8 bytes");

// -----------------------------------------------------------------------------
// Hello (CMD_HELLO). Same struct in both directions; the server's hello is
// the response, so the client validates `version_major` and required
// feature bits before any further traffic.
// -----------------------------------------------------------------------------

struct MsgHello {
    uint8_t  version_major;  // GGML_TB_RDMA_PROTO_VERSION_MAJOR
    uint8_t  version_minor;  // informational
    uint16_t reserved;       // must be zero
    uint32_t features;       // bit-or of GGML_TB_RDMA_FEATURE_*
    uint32_t required;       // subset of features the peer requires to proceed
    uint32_t reserved2;      // must be zero
};
static_assert(sizeof(MsgHello) == 16, "MsgHello must be 16 bytes");

// -----------------------------------------------------------------------------
// Tensor on the wire (matches ggml-rpc — ported here so we can drop the
// `WireTensor` from the deleted backend).
// -----------------------------------------------------------------------------

struct WireTensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint64_t ne[MAX_DIMS];
    uint64_t nb[MAX_DIMS];
    uint32_t op;
    int32_t  op_params[MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char     name[MAX_NAME];
    char     padding[4];
};
static_assert(sizeof(WireTensor) % 8 == 0, "WireTensor must be 8-byte aligned");

// -----------------------------------------------------------------------------
// Per-command payload structs (fixed-size unless noted)
// -----------------------------------------------------------------------------

// CMD_ALLOC_BUFFER
struct MsgAllocBufferReq {
    uint64_t size;
};
struct MsgAllocBufferRsp {
    uint64_t remote_ptr;     // 0 on failure
    uint64_t remote_size;
};

// CMD_FREE_BUFFER
struct MsgFreeBufferReq {
    uint64_t remote_ptr;
};
struct MsgFreeBufferRsp {
    uint8_t  ok;
    uint8_t  reserved[7];
};

// CMD_BUFFER_GET_BASE
struct MsgBufferGetBaseReq {
    uint64_t remote_ptr;
};
struct MsgBufferGetBaseRsp {
    uint64_t base_ptr;
};

// CMD_BUFFER_CLEAR
struct MsgBufferClearReq {
    uint64_t remote_ptr;
    uint8_t  value;
    uint8_t  reserved[7];
};
struct MsgBufferClearRsp {
    uint8_t  ok;
    uint8_t  reserved[7];
};

// CMD_BUFFER_MEMSET (per-tensor memset; M1.4)
struct MsgBufferMemsetReq {
    uint64_t remote_ptr;     // owning buffer
    uint64_t tensor_id;      // tensor handle in the owning buffer
    uint64_t offset;
    uint64_t size;
    uint8_t  value;
    uint8_t  reserved[7];
};
struct MsgBufferMemsetRsp {
    uint8_t  ok;
    uint8_t  reserved[7];
};

// CMD_SET_TENSOR — variable: MsgSetTensorReq followed by `size` data bytes.
struct MsgSetTensorReq {
    WireTensor tensor;
    uint64_t offset;
    uint64_t size;
    // followed by `size` bytes of tensor data
};
struct MsgSetTensorRsp {
    uint8_t  ok;
    uint8_t  reserved[7];
};

// CMD_GET_TENSOR — request fixed, response variable (MsgGetTensorRsp +
// `size` data bytes).
struct MsgGetTensorReq {
    WireTensor tensor;
    uint64_t offset;
    uint64_t size;
};
struct MsgGetTensorRsp {
    uint64_t size;
    // followed by `size` bytes of tensor data
};

// CMD_COPY_TENSOR
struct MsgCopyTensorReq {
    WireTensor src;
    WireTensor dst;
};
struct MsgCopyTensorRsp {
    uint8_t  ok;
    uint8_t  reserved[7];
};

// CMD_INIT_TENSOR
struct MsgInitTensorReq {
    WireTensor tensor;
};
struct MsgInitTensorRsp {
    uint8_t  ok;
    uint8_t  reserved[7];
};

// CMD_GRAPH_COMPUTE — variable:
//   MsgGraphComputeReq
//   + n_nodes   * uint64_t  (node ids)
//   + n_tensors * WireTensor
struct MsgGraphComputeReq {
    uint32_t n_nodes;
    uint32_t n_tensors;
};
struct MsgGraphComputeRsp {
    uint8_t  status;   // ggml_status
    uint8_t  reserved[7];
};

// CMD_GET_DEVICE_MEMORY — no request payload
struct MsgGetDeviceMemoryRsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

// CMD_GET_ALIGNMENT — no request payload
struct MsgGetAlignmentRsp {
    uint64_t alignment;
};

// CMD_GET_MAX_SIZE — no request payload
struct MsgGetMaxSizeRsp {
    uint64_t max_size;
};

// -----------------------------------------------------------------------------
// RDMA QP destination exchange (M2a) — sent over TCP after hello.
// -----------------------------------------------------------------------------

// 16-byte GID, sent as raw bytes (network order matches verbs union ibv_gid).
struct MsgGid {
    uint8_t  raw[16];
};
static_assert(sizeof(MsgGid) == 16, "MsgGid must be 16 bytes");

// QP destination + path attributes that both sides need before they can
// transition INIT → RTR → RTS.
struct MsgQPInfo {
    uint32_t qp_num;          // remote QP number
    uint32_t recv_credits;    // R: number of pre-posted recv slots on peer
    uint32_t frame_size;      // chosen RDMA frame size (path MTU bytes)
    uint8_t  port_num;        // IB port number on peer (usually 1)
    uint8_t  gid_index;       // GID table index used
    uint8_t  path_mtu;        // enum ibv_mtu value (e.g. IBV_MTU_1024 == 3)
    uint8_t  reserved;
    uint32_t psn;             // initial packet sequence number
    MsgGid   gid;             // 16-byte GID
    uint16_t lid;             // peer port LID (from ibv_query_port, used as ah_attr.dlid)
    uint16_t reserved2;       // must be zero
};
static_assert(sizeof(MsgQPInfo) == 40, "MsgQPInfo must be 40 bytes");

// CMD_RDMA_READY — server -> client first, then echoed back over RDMA via
// the readiness ping. Carries a tiny random nonce so both sides can verify
// that the RDMA path round-tripped the right bytes.
struct MsgRdmaReady {
    uint64_t nonce;
};
static_assert(sizeof(MsgRdmaReady) == 8, "MsgRdmaReady must be 8 bytes");

// -----------------------------------------------------------------------------
// RDMA inner frame format (M2b/M2c).
//
// Every RDMA SEND posts one slot whose first `sizeof(RdmaFrame)` bytes are
// the header below; the remainder is up to `payload_per_frame` bytes of
// payload. Frame total size is always `frame_size_bytes` == path MTU.
// -----------------------------------------------------------------------------

enum RdmaFrameKind : uint8_t {
    // Reliability-only frames (no application payload):
    RDMA_KIND_FLOW    = 0,  // ACK-only / credit return
    // Control messages (CONTROL frames). `payload_offset_or_total` holds
    // total length on FIRST; subsequent CTRL frames carry chunks of the
    // already-known total (sender packs the same `msg_id` across them).
    RDMA_KIND_CTRL    = 1,
    // Data messages (DATA frames). `payload_offset_or_total` carries
    // `total` on FIRST and `dst_off` on subsequent DATA frames.
    RDMA_KIND_DATA    = 2,
};

enum RdmaFrameFlags : uint8_t {
    RDMA_FLAG_FIRST   = 1 << 0,  // first frame of a logical message
    RDMA_FLAG_LAST    = 1 << 1,  // last frame of a logical message
};

struct RdmaFrame {
    uint32_t msg_id;     // logical message id (monotonic per direction)
    uint32_t seq;        // per-direction frame sequence number
    uint32_t ack;        // cumulative ack of the *other* direction's seq
    uint8_t  kind;       // RdmaFrameKind
    uint8_t  flags;      // bitmask of RdmaFrameFlags
    uint16_t len;        // payload bytes in this frame (after header)
    uint64_t payload_offset_or_total; // see RdmaFrameKind
};
static_assert(sizeof(RdmaFrame) == 24, "RdmaFrame must be 24 bytes");

// Defaults for the reliability layer. Tunable later via env vars.
constexpr uint32_t TB_RDMA_DEFAULT_RECV_CREDITS = 64;
constexpr uint32_t TB_RDMA_DEFAULT_SEND_WINDOW  = 8;
constexpr int      TB_RDMA_RETRANSMIT_TIMEOUT_MS = 50;
constexpr int      TB_RDMA_ACK_DELAY_MS          = 1;
constexpr int      TB_RDMA_MAX_RETRANSMITS       = 3;
constexpr int      TB_RDMA_DEFAULT_OP_TIMEOUT_MS = 30 * 1000;

} // namespace ggml_tb_rdma

#pragma pack(pop)

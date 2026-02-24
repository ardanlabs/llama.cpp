#include "server.h"

#include "endpoint.h"
#include "protocol.h"
#include "socket.h"
#include "transport.h"
#include "validation.h"

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <atomic>
#include <cinttypes>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ggml_tb_rdma {

// SIGINT/SIGTERM handler — set the flag, close the listen socket from the
// main loop. Single-process scope; we don't multiplex servers.
static std::atomic<bool> g_stop{false};
static int               g_listen_fd = -1;
static void on_signal(int /*sig*/) {
    g_stop.store(true);
    if (g_listen_fd >= 0) {
        tcp_close(g_listen_fd);
        g_listen_fd = -1;
    }
}

// -----------------------------------------------------------------------------
// Per-connection server state
// -----------------------------------------------------------------------------

struct ServerSession {
    Transport       tx;
    ggml_backend_t  backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;

    // remote_ptr → owning buffer (the `remote_ptr` is the buffer pointer cast
    // to uint64_t and is what the client sees).
    std::unordered_map<uint64_t, ggml_backend_buffer_t> buffers;

    // Scratch buffer reused across graph_compute calls.
    std::vector<uint8_t> graph_scratch;

    bool valid_buffer(uint64_t remote_ptr) const {
        return buffers.find(remote_ptr) != buffers.end();
    }

    ~ServerSession() {
        for (auto & kv : buffers) {
            ggml_backend_buffer_free(kv.second);
        }
    }
};

// -----------------------------------------------------------------------------
// WireTensor → ggml_tensor
// -----------------------------------------------------------------------------

static ggml_tensor * deserialize_tensor(ServerSession & s,
                                        ggml_context * ctx,
                                        const WireTensor & wt) {
    if (wt.type >= GGML_TYPE_COUNT)               return nullptr;
    if (ggml_blck_size((ggml_type) wt.type) == 0) return nullptr;

    ggml_tensor * t = ggml_new_tensor_4d(ctx, (ggml_type) wt.type,
                                          wt.ne[0], wt.ne[1], wt.ne[2], wt.ne[3]);
    if (!t) return nullptr;

    for (int i = 0; i < GGML_MAX_DIMS; i++) t->nb[i] = wt.nb[i];

    t->buffer = reinterpret_cast<ggml_backend_buffer_t>(wt.buffer);
    if (t->buffer && s.buffers.find(reinterpret_cast<uint64_t>(t->buffer)) == s.buffers.end()) {
        t->buffer = nullptr;
    }

    if (t->buffer) {
        const uint64_t tensor_size = (uint64_t) ggml_nbytes(t);
        const uint64_t buf_start   = (uint64_t) ggml_backend_buffer_get_base(t->buffer);
        const uint64_t buf_size    = (uint64_t) ggml_backend_buffer_get_size(t->buffer);
        if (wt.data + tensor_size < wt.data) return nullptr; // overflow
        if (wt.data < buf_start)                                return nullptr;
        if (wt.data + tensor_size > buf_start + buf_size)       return nullptr;
    }

    t->op = (ggml_op) wt.op;
    std::memcpy(t->op_params, wt.op_params, sizeof(t->op_params));
    t->flags = wt.flags;
    t->data  = reinterpret_cast<void *>(wt.data);
    ggml_set_name(t, wt.name);
    return t;
}

static ggml_tensor * create_node(ServerSession & s,
                                 uint64_t id,
                                 ggml_context * ctx,
                                 const std::unordered_map<uint64_t, const WireTensor *> & ptrs,
                                 std::unordered_map<uint64_t, ggml_tensor *> & built) {
    if (id == 0) return nullptr;
    auto it = built.find(id);
    if (it != built.end()) return it->second;

    auto pit = ptrs.find(id);
    if (pit == ptrs.end()) return nullptr;
    const WireTensor & wt = *pit->second;

    ggml_tensor * t = deserialize_tensor(s, ctx, wt);
    if (!t) return nullptr;
    if (t->buffer == nullptr && t->data != nullptr) return nullptr;
    built[id] = t;

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (wt.src[i] == 0) { t->src[i] = nullptr; continue; }
        t->src[i] = create_node(s, wt.src[i], ctx, ptrs, built);
        if (!t->src[i]) return nullptr;
    }
    if (wt.view_src == 0) {
        t->view_src = nullptr;
    } else {
        t->view_src = create_node(s, wt.view_src, ctx, ptrs, built);
        if (!t->view_src) return nullptr;
    }
    t->view_offs = wt.view_offs;
    return t;
}

// -----------------------------------------------------------------------------
// Command handlers — each returns true on a normal reply (which it sent
// itself), or false if the transport already moved to the error state.
// -----------------------------------------------------------------------------

static bool reply_error(ServerSession & s, TbRdmaError code, const char * msg) {
    s.tx.send_error_and_close(code, msg ? msg : "");
    return false;
}

static bool handle_alloc_buffer(ServerSession & s, const std::vector<uint8_t> & in) {
    if (in.size() != sizeof(MsgAllocBufferReq)) return reply_error(s, ERR_PROTOCOL, "alloc_buffer: bad size");
    MsgAllocBufferReq req;
    std::memcpy(&req, in.data(), sizeof(req));

    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(s.buft, req.size);
    MsgAllocBufferRsp rsp = {};
    if (buf) {
        rsp.remote_ptr  = reinterpret_cast<uint64_t>(buf);
        rsp.remote_size = ggml_backend_buffer_get_size(buf);
        s.buffers[rsp.remote_ptr] = buf;
    }
    return s.tx.send_response(CMD_ALLOC_BUFFER, rsp) == RequestStatus::Ok;
}

static bool handle_free_buffer(ServerSession & s, const std::vector<uint8_t> & in) {
    if (in.size() != sizeof(MsgFreeBufferReq)) return reply_error(s, ERR_PROTOCOL, "free_buffer: bad size");
    MsgFreeBufferReq req;
    std::memcpy(&req, in.data(), sizeof(req));

    MsgFreeBufferRsp rsp = {};
    auto it = s.buffers.find(req.remote_ptr);
    if (it != s.buffers.end()) {
        ggml_backend_buffer_free(it->second);
        s.buffers.erase(it);
        rsp.ok = 1;
    }
    return s.tx.send_response(CMD_FREE_BUFFER, rsp) == RequestStatus::Ok;
}

static bool handle_buffer_get_base(ServerSession & s, const std::vector<uint8_t> & in) {
    if (in.size() != sizeof(MsgBufferGetBaseReq)) return reply_error(s, ERR_PROTOCOL, "get_base: bad size");
    MsgBufferGetBaseReq req;
    std::memcpy(&req, in.data(), sizeof(req));

    MsgBufferGetBaseRsp rsp = {};
    auto it = s.buffers.find(req.remote_ptr);
    if (it != s.buffers.end()) {
        rsp.base_ptr = reinterpret_cast<uint64_t>(ggml_backend_buffer_get_base(it->second));
    }
    return s.tx.send_response(CMD_BUFFER_GET_BASE, rsp) == RequestStatus::Ok;
}

static bool handle_buffer_clear(ServerSession & s, const std::vector<uint8_t> & in) {
    if (in.size() != sizeof(MsgBufferClearReq)) return reply_error(s, ERR_PROTOCOL, "clear: bad size");
    MsgBufferClearReq req;
    std::memcpy(&req, in.data(), sizeof(req));

    MsgBufferClearRsp rsp = {};
    auto it = s.buffers.find(req.remote_ptr);
    if (it != s.buffers.end()) {
        ggml_backend_buffer_clear(it->second, req.value);
        rsp.ok = 1;
    }
    return s.tx.send_response(CMD_BUFFER_CLEAR, rsp) == RequestStatus::Ok;
}

static bool handle_buffer_memset(ServerSession & s, const std::vector<uint8_t> & in) {
    if (in.size() != sizeof(MsgBufferMemsetReq)) return reply_error(s, ERR_PROTOCOL, "memset: bad size");
    MsgBufferMemsetReq req;
    std::memcpy(&req, in.data(), sizeof(req));

    MsgBufferMemsetRsp rsp = {};
    auto it = s.buffers.find(req.remote_ptr);
    if (it == s.buffers.end()) {
        return s.tx.send_response(CMD_BUFFER_MEMSET, rsp) == RequestStatus::Ok;
    }
    auto * tensor = reinterpret_cast<ggml_tensor *>(req.tensor_id);
    if (!tensor) {
        return s.tx.send_response(CMD_BUFFER_MEMSET, rsp) == RequestStatus::Ok;
    }
    const uint64_t tensor_size = (uint64_t) ggml_nbytes(tensor);
    if (!validate_tensor_range(tensor_size, req.offset, req.size)) {
        return s.tx.send_response(CMD_BUFFER_MEMSET, rsp) == RequestStatus::Ok;
    }
    ggml_backend_tensor_memset(tensor, req.value, req.offset, req.size);
    rsp.ok = 1;
    return s.tx.send_response(CMD_BUFFER_MEMSET, rsp) == RequestStatus::Ok;
}

static bool handle_set_tensor(ServerSession & s, const std::vector<uint8_t> & in) {
    if (in.size() < sizeof(MsgSetTensorReq)) return reply_error(s, ERR_PROTOCOL, "set_tensor: bad size");
    MsgSetTensorReq req;
    std::memcpy(&req, in.data(), sizeof(req));

    if (req.size > MAX_TENSOR_BYTES) return reply_error(s, ERR_INVALID_ARGUMENT, "set_tensor: oversize");

    // After M2c the payload can take one of two shapes:
    //   TCP path:  [MsgSetTensorReq][body]
    //   RDMA path: [MsgSetTensorReq] only — body arrives via recv_bulk_body.
    const bool body_inline = (in.size() == sizeof(MsgSetTensorReq) + req.size);
    const bool body_separate = (in.size() == sizeof(MsgSetTensorReq));
    if (!body_inline && !body_separate) {
        return reply_error(s, ERR_PROTOCOL, "set_tensor: payload mismatch");
    }

    auto * tensor = reinterpret_cast<ggml_tensor *>(req.tensor.id);
    MsgSetTensorRsp rsp = {};
    if (!tensor || !tensor->buffer) {
        // Still must drain the body so the stream stays in sync.
        if (body_separate && req.size > 0) {
            std::vector<uint8_t> scratch(req.size);
            (void) s.tx.recv_bulk_body(scratch.data(), req.size, nullptr);
        }
        return s.tx.send_response(CMD_SET_TENSOR, rsp) == RequestStatus::Ok;
    }
    const uint64_t tensor_size = (uint64_t) ggml_nbytes(tensor);
    if (!validate_tensor_range(tensor_size, req.offset, req.size)) {
        if (body_separate && req.size > 0) {
            std::vector<uint8_t> scratch(req.size);
            (void) s.tx.recv_bulk_body(scratch.data(), req.size, nullptr);
        }
        return s.tx.send_response(CMD_SET_TENSOR, rsp) == RequestStatus::Ok;
    }

    // For the RDMA path, stream directly into a temporary buffer so we can
    // hand it to the backend. (Future optimisation: stream straight into the
    // backend buffer when it exposes a raw host pointer.)
    if (body_separate) {
        std::vector<uint8_t> body(req.size);
        if (req.size > 0) {
            RequestStatus st = s.tx.recv_bulk_body(body.data(), req.size, nullptr);
            if (st != RequestStatus::Ok) return false;
        }
        ggml_backend_tensor_set(tensor, body.data(), req.offset, req.size);
    } else {
        ggml_backend_tensor_set(tensor, in.data() + sizeof(MsgSetTensorReq), req.offset, req.size);
    }
    rsp.ok = 1;
    return s.tx.send_response(CMD_SET_TENSOR, rsp) == RequestStatus::Ok;
}

static bool handle_get_tensor(ServerSession & s, const std::vector<uint8_t> & in) {
    if (in.size() != sizeof(MsgGetTensorReq)) return reply_error(s, ERR_PROTOCOL, "get_tensor: bad size");
    MsgGetTensorReq req;
    std::memcpy(&req, in.data(), sizeof(req));

    if (req.size > MAX_TENSOR_BYTES) return reply_error(s, ERR_INVALID_ARGUMENT, "get_tensor: oversize");

    auto * tensor = reinterpret_cast<ggml_tensor *>(req.tensor.id);
    if (!tensor || !tensor->buffer) {
        return reply_error(s, ERR_BAD_TENSOR, "get_tensor: unknown tensor");
    }
    const uint64_t tensor_size = (uint64_t) ggml_nbytes(tensor);
    if (!validate_tensor_range(tensor_size, req.offset, req.size)) {
        return reply_error(s, ERR_INVALID_ARGUMENT, "get_tensor: out of range");
    }

    std::vector<uint8_t> data(req.size);
    ggml_backend_tensor_get(tensor, data.data(), req.offset, req.size);

    MsgGetTensorRsp rsp = {};
    rsp.size = req.size;
    return s.tx.send_response_with_body(CMD_GET_TENSOR,
                                        &rsp, sizeof(rsp),
                                        data.data(), data.size()) == RequestStatus::Ok;
}

static bool handle_copy_tensor(ServerSession & s, const std::vector<uint8_t> & in) {
    if (in.size() != sizeof(MsgCopyTensorReq)) return reply_error(s, ERR_PROTOCOL, "copy_tensor: bad size");
    MsgCopyTensorReq req;
    std::memcpy(&req, in.data(), sizeof(req));

    MsgCopyTensorRsp rsp = {};
    auto * src = reinterpret_cast<ggml_tensor *>(req.src.id);
    auto * dst = reinterpret_cast<ggml_tensor *>(req.dst.id);
    if (src && dst && src->buffer && dst->buffer) {
        rsp.ok = ggml_backend_buffer_copy_tensor(src, dst) ? 1 : 0;
    }
    return s.tx.send_response(CMD_COPY_TENSOR, rsp) == RequestStatus::Ok;
}

static bool handle_init_tensor(ServerSession & s, const std::vector<uint8_t> & in) {
    if (in.size() != sizeof(MsgInitTensorReq)) return reply_error(s, ERR_PROTOCOL, "init_tensor: bad size");
    MsgInitTensorReq req;
    std::memcpy(&req, in.data(), sizeof(req));

    MsgInitTensorRsp rsp = {};
    auto * tensor = reinterpret_cast<ggml_tensor *>(req.tensor.id);
    if (tensor && tensor->buffer) {
        ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
        rsp.ok = 1;
    }
    return s.tx.send_response(CMD_INIT_TENSOR, rsp) == RequestStatus::Ok;
}

static bool handle_graph_compute(ServerSession & s, const std::vector<uint8_t> & in) {
    if (in.size() < sizeof(MsgGraphComputeReq)) return reply_error(s, ERR_PROTOCOL, "graph: bad size");
    MsgGraphComputeReq req;
    std::memcpy(&req, in.data(), sizeof(req));

    if (req.n_nodes   > MAX_GRAPH_NODES)   return reply_error(s, ERR_INVALID_ARGUMENT, "graph: too many nodes");
    if (req.n_tensors > MAX_GRAPH_TENSORS) return reply_error(s, ERR_INVALID_ARGUMENT, "graph: too many tensors");

    uint64_t expected = sizeof(MsgGraphComputeReq);
    uint64_t nodes_bytes   = 0;
    uint64_t tensors_bytes = 0;
    if (!checked_mul_u64(req.n_nodes,   sizeof(uint64_t),   &nodes_bytes)   ||
        !checked_mul_u64(req.n_tensors, sizeof(WireTensor), &tensors_bytes) ||
        !checked_add_u64(nodes_bytes,   &expected) ||
        !checked_add_u64(tensors_bytes, &expected)) {
        return reply_error(s, ERR_INVALID_ARGUMENT, "graph: overflow");
    }
    if (expected != in.size()) return reply_error(s, ERR_PROTOCOL, "graph: payload mismatch");

    const uint8_t * p = in.data() + sizeof(MsgGraphComputeReq);
    const uint64_t   * node_ids = reinterpret_cast<const uint64_t *>(p);
    const WireTensor * wtensors = reinterpret_cast<const WireTensor *>(p + nodes_bytes);

    const size_t buf_size = ggml_tensor_overhead() * (req.n_nodes + req.n_tensors)
                          + ggml_graph_overhead_custom(req.n_nodes, false);
    if (s.graph_scratch.size() < buf_size) s.graph_scratch.resize(buf_size);

    ggml_init_params params = {};
    params.mem_size   = buf_size;
    params.mem_buffer = s.graph_scratch.data();
    params.no_alloc   = true;

    ggml_context * ctx = ggml_init(params);
    if (!ctx) return reply_error(s, ERR_OOM, "graph: ggml_init failed");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, req.n_nodes, false);
    graph->n_nodes = static_cast<int>(req.n_nodes);

    std::unordered_map<uint64_t, const WireTensor *> ptrs;
    ptrs.reserve(req.n_tensors);
    for (uint32_t i = 0; i < req.n_tensors; i++) ptrs.emplace(wtensors[i].id, &wtensors[i]);

    std::unordered_map<uint64_t, ggml_tensor *> built;
    built.reserve(req.n_nodes);

    bool ok = true;
    for (uint32_t i = 0; i < req.n_nodes; i++) {
        const uint64_t id = node_ids[i];
        ggml_tensor * t = create_node(s, id, ctx, ptrs, built);
        graph->nodes[i] = t;
        if (!t && id != 0) { ok = false; break; }
    }

    MsgGraphComputeRsp rsp = {};
    if (!ok) {
        rsp.status = static_cast<uint8_t>(GGML_STATUS_FAILED);
    } else {
        rsp.status = static_cast<uint8_t>(ggml_backend_graph_compute(s.backend, graph));
    }
    ggml_free(ctx);
    return s.tx.send_response(CMD_GRAPH_COMPUTE, rsp) == RequestStatus::Ok;
}

static bool handle_get_alignment(ServerSession & s) {
    MsgGetAlignmentRsp rsp = {};
    rsp.alignment = ggml_backend_buft_get_alignment(s.buft);
    return s.tx.send_response(CMD_GET_ALIGNMENT, rsp) == RequestStatus::Ok;
}

static bool handle_get_max_size(ServerSession & s) {
    MsgGetMaxSizeRsp rsp = {};
    rsp.max_size = ggml_backend_buft_get_max_size(s.buft);
    return s.tx.send_response(CMD_GET_MAX_SIZE, rsp) == RequestStatus::Ok;
}

static bool handle_get_device_memory(ServerSession & s) {
    MsgGetDeviceMemoryRsp rsp = {};
    ggml_backend_dev_t dev = ggml_backend_get_device(s.backend);
    if (dev) {
        size_t free_mem = 0, total_mem = 0;
        ggml_backend_dev_memory(dev, &free_mem, &total_mem);
        rsp.free_mem  = free_mem;
        rsp.total_mem = total_mem;
    }
    return s.tx.send_response(CMD_GET_DEVICE_MEMORY, rsp) == RequestStatus::Ok;
}

// -----------------------------------------------------------------------------
// Per-connection dispatch
// -----------------------------------------------------------------------------

static void serve_one(int conn_fd, const std::string & rdma_device, ggml_backend_t backend) {
    ServerSession s;
    s.backend = backend;
    s.buft    = ggml_backend_get_default_buffer_type(backend);

    std::string err;
    if (!s.tx.server_accept(conn_fd, rdma_device, &err)) {
        std::fprintf(stderr, "[tb-rdma] handshake failed: %s\n", err.c_str());
        return;
    }
    std::fprintf(stderr, "[tb-rdma] client connected\n");

    std::vector<uint8_t> payload;
    while (true) {
        MsgHeader hdr = {};
        const uint32_t cap = (hdr.cmd == CMD_GRAPH_COMPUTE) ? MAX_GRAPH_FRAME_BYTES
                                                            : MAX_CONTROL_FRAME_BYTES;
        RequestStatus st = s.tx.recv_request(&hdr, &payload, cap);
        if (st == RequestStatus::Disconnected || st == RequestStatus::Closed) break;
        if (st != RequestStatus::Ok) {
            std::fprintf(stderr, "[tb-rdma] transport error: %d\n", (int) st);
            break;
        }

        // Apply the per-command frame cap *after* reading the header so that
        // graph_compute (larger cap) is permitted only for its own command.
        if (hdr.cmd == CMD_GRAPH_COMPUTE) {
            if (hdr.payload_size > MAX_GRAPH_FRAME_BYTES) {
                reply_error(s, ERR_PROTOCOL, "graph: oversized");
                break;
            }
        } else {
            if (hdr.payload_size > MAX_CONTROL_FRAME_BYTES) {
                reply_error(s, ERR_PROTOCOL, "frame: oversized");
                break;
            }
        }

        bool keep_going = true;
        switch (hdr.cmd) {
            case CMD_GOODBYE:           keep_going = false; break;
            case CMD_ALLOC_BUFFER:      keep_going = handle_alloc_buffer(s, payload); break;
            case CMD_FREE_BUFFER:       keep_going = handle_free_buffer(s, payload); break;
            case CMD_BUFFER_GET_BASE:   keep_going = handle_buffer_get_base(s, payload); break;
            case CMD_BUFFER_CLEAR:      keep_going = handle_buffer_clear(s, payload); break;
            case CMD_BUFFER_MEMSET:     keep_going = handle_buffer_memset(s, payload); break;
            case CMD_SET_TENSOR:        keep_going = handle_set_tensor(s, payload); break;
            case CMD_GET_TENSOR:        keep_going = handle_get_tensor(s, payload); break;
            case CMD_COPY_TENSOR:       keep_going = handle_copy_tensor(s, payload); break;
            case CMD_INIT_TENSOR:       keep_going = handle_init_tensor(s, payload); break;
            case CMD_GRAPH_COMPUTE:     keep_going = handle_graph_compute(s, payload); break;
            case CMD_GET_DEVICE_MEMORY: keep_going = handle_get_device_memory(s); break;
            case CMD_GET_ALIGNMENT:     keep_going = handle_get_alignment(s); break;
            case CMD_GET_MAX_SIZE:      keep_going = handle_get_max_size(s); break;
            default:
                reply_error(s, ERR_UNSUPPORTED, "unknown cmd");
                keep_going = false;
                break;
        }
        if (!keep_going) break;
    }
    std::fprintf(stderr, "[tb-rdma] client disconnected\n");
}

// -----------------------------------------------------------------------------
// Server loop
// -----------------------------------------------------------------------------

void run_server(const std::string & endpoint_str,
                const std::string & rdma_device,
                ggml_backend_t      backend) {
    Endpoint ep;
    std::string err;
    if (!parse_endpoint(endpoint_str, ep, &err)) {
        std::fprintf(stderr, "[tb-rdma] bad endpoint '%s': %s\n", endpoint_str.c_str(), err.c_str());
        return;
    }

    g_listen_fd = tcp_listen(ep, &err);
    if (g_listen_fd < 0) {
        std::fprintf(stderr, "[tb-rdma] %s\n", err.c_str());
        return;
    }
    std::fprintf(stderr, "[tb-rdma] listening on %s:%u\n", ep.host.c_str(), ep.port);

    // Per-socket SIGPIPE is suppressed via SO_NOSIGPIPE; install only the
    // SIGINT/SIGTERM hooks here (no process-global SIG_IGN of SIGPIPE).
    struct sigaction sa = {};
    sa.sa_handler = on_signal;
    sigaction(SIGINT,  &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    while (!g_stop.load()) {
        int conn_fd = tcp_accept(g_listen_fd, &err);
        if (conn_fd < 0) {
            if (g_stop.load()) break;
            std::fprintf(stderr, "[tb-rdma] accept: %s\n", err.c_str());
            continue;
        }
        serve_one(conn_fd, rdma_device, backend);
        // serve_one's ServerSession destructor closes the connection.
    }

    if (g_listen_fd >= 0) {
        tcp_close(g_listen_fd);
        g_listen_fd = -1;
    }
}

} // namespace ggml_tb_rdma

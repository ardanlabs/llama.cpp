#include "ggml-tb-rdma.h"

#include "endpoint.h"
#include "ibv.h"
#include "protocol.h"
#include "server.h"
#include "transport.h"
#include "validation.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace ggml_tb_rdma;

// -----------------------------------------------------------------------------
// Per-endpoint shared client context — created once on the first call to
// either `ggml_backend_tb_rdma_init` or `ggml_backend_tb_rdma_buffer_type`
// for a given endpoint string. M1.3.
// -----------------------------------------------------------------------------

struct EndpointContext {
    std::string                endpoint;
    std::string                rdma_device;
    Transport                  transport;
    ggml_backend_buffer_type   buft = {};
    bool                       buft_inited = false;
    std::mutex                 mu;  // protects `transport` lifecycle + buft init
};

static std::mutex g_registry_mu;
static std::unordered_map<std::string, std::shared_ptr<EndpointContext>> g_registry;

static std::shared_ptr<EndpointContext> get_or_create_endpoint(const std::string & endpoint) {
    std::lock_guard<std::mutex> lock(g_registry_mu);
    auto it = g_registry.find(endpoint);
    if (it != g_registry.end()) return it->second;
    auto p = std::make_shared<EndpointContext>();
    p->endpoint = endpoint;
    g_registry.emplace(endpoint, p);
    return p;
}

// -----------------------------------------------------------------------------
// Tensor serialization (client side)
// -----------------------------------------------------------------------------

static WireTensor serialize_tensor(const ggml_tensor * t) {
    WireTensor wt = {};
    wt.id     = reinterpret_cast<uint64_t>(t);
    wt.type   = t->type;
    wt.buffer = t->buffer ? reinterpret_cast<uint64_t>(t->buffer) : 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        wt.ne[i] = static_cast<uint64_t>(t->ne[i]);
        wt.nb[i] = static_cast<uint64_t>(t->nb[i]);
    }
    wt.op = t->op;
    std::memcpy(wt.op_params, t->op_params, sizeof(wt.op_params));
    wt.flags = t->flags;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        wt.src[i] = t->src[i] ? reinterpret_cast<uint64_t>(t->src[i]) : 0;
    }
    wt.view_src  = t->view_src ? reinterpret_cast<uint64_t>(t->view_src) : 0;
    wt.view_offs = t->view_offs;
    wt.data      = reinterpret_cast<uint64_t>(t->data);
    if (t->name[0]) std::strncpy(wt.name, t->name, sizeof(wt.name) - 1);
    return wt;
}

// -----------------------------------------------------------------------------
// Buffer context — buffer = remote handle + back-pointer to its endpoint.
// We deliberately do NOT mirror buffer contents locally (M1.6).
// -----------------------------------------------------------------------------

struct BufferContext {
    std::shared_ptr<EndpointContext> ep;
    uint64_t                         remote_ptr = 0;
    uint64_t                         remote_size = 0;
};

// -----------------------------------------------------------------------------
// Buffer interface
// -----------------------------------------------------------------------------

static void tb_buffer_free(ggml_backend_buffer_t buffer) {
    auto * bc = static_cast<BufferContext *>(buffer->context);
    MsgFreeBufferReq req = { bc->remote_ptr };
    MsgFreeBufferRsp rsp = {};
    (void) bc->ep->transport.request(CMD_FREE_BUFFER, req, CMD_FREE_BUFFER, rsp);
    delete bc;
}

static void * tb_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * bc = static_cast<BufferContext *>(buffer->context);
    MsgBufferGetBaseReq req = { bc->remote_ptr };
    MsgBufferGetBaseRsp rsp = {};
    if (bc->ep->transport.request(CMD_BUFFER_GET_BASE, req, CMD_BUFFER_GET_BASE, rsp) != RequestStatus::Ok) {
        return nullptr;
    }
    return reinterpret_cast<void *>(rsp.base_ptr);
}

static enum ggml_status tb_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto * bc = static_cast<BufferContext *>(buffer->context);
    MsgInitTensorReq req = {};
    req.tensor = serialize_tensor(tensor);
    MsgInitTensorRsp rsp = {};
    if (bc->ep->transport.request(CMD_INIT_TENSOR, req, CMD_INIT_TENSOR, rsp) != RequestStatus::Ok) {
        return GGML_STATUS_FAILED;
    }
    return rsp.ok ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

static void tb_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                 const void * data, size_t offset, size_t size) {
    auto * bc = static_cast<BufferContext *>(buffer->context);
    MsgSetTensorReq req = {};
    req.tensor = serialize_tensor(tensor);
    req.offset = offset;
    req.size   = size;

    MsgSetTensorRsp rsp = {};
    RequestStatus st = bc->ep->transport.request_bulk_send(
        CMD_SET_TENSOR,
        &req, sizeof(req),
        data, size,
        CMD_SET_TENSOR, &rsp, sizeof(rsp));
    if (st != RequestStatus::Ok) {
        std::fprintf(stderr, "[tb-rdma] set_tensor failed (status=%d)\n", (int) st);
        return;
    }
    if (!rsp.ok) std::fprintf(stderr, "[tb-rdma] set_tensor: server reported failure\n");
}

static void tb_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
                                 void * data, size_t offset, size_t size) {
    auto * bc = static_cast<BufferContext *>(buffer->context);
    MsgGetTensorReq req = {};
    req.tensor = serialize_tensor(tensor);
    req.offset = offset;
    req.size   = size;

    MsgGetTensorRsp rsp = {};
    RequestStatus st = bc->ep->transport.request_bulk_recv(
        CMD_GET_TENSOR,
        &req, sizeof(req),
        CMD_GET_TENSOR, &rsp, sizeof(rsp),
        data, size);
    if (st != RequestStatus::Ok) {
        std::fprintf(stderr, "[tb-rdma] get_tensor failed (status=%d)\n", (int) st);
        return;
    }
    if (rsp.size != size) {
        std::fprintf(stderr, "[tb-rdma] get_tensor: size mismatch (got %llu, want %zu)\n",
                     (unsigned long long) rsp.size, size);
    }
}

static bool tb_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    auto * bc = static_cast<BufferContext *>(buffer->context);
    MsgCopyTensorReq req = {};
    req.src = serialize_tensor(src);
    req.dst = serialize_tensor(dst);
    MsgCopyTensorRsp rsp = {};
    if (bc->ep->transport.request(CMD_COPY_TENSOR, req, CMD_COPY_TENSOR, rsp) != RequestStatus::Ok) {
        return false;
    }
    return rsp.ok != 0;
}

static void tb_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * bc = static_cast<BufferContext *>(buffer->context);
    MsgBufferClearReq req = {};
    req.remote_ptr = bc->remote_ptr;
    req.value      = value;
    MsgBufferClearRsp rsp = {};
    (void) bc->ep->transport.request(CMD_BUFFER_CLEAR, req, CMD_BUFFER_CLEAR, rsp);
}

static void tb_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                    uint8_t value, size_t offset, size_t size) {
    auto * bc = static_cast<BufferContext *>(buffer->context);
    MsgBufferMemsetReq req = {};
    req.remote_ptr = bc->remote_ptr;
    req.tensor_id  = reinterpret_cast<uint64_t>(tensor);
    req.offset     = offset;
    req.size       = size;
    req.value      = value;
    MsgBufferMemsetRsp rsp = {};
    if (bc->ep->transport.request(CMD_BUFFER_MEMSET, req, CMD_BUFFER_MEMSET, rsp) != RequestStatus::Ok || !rsp.ok) {
        std::fprintf(stderr, "[tb-rdma] memset_tensor: server reported failure\n");
    }
}

static ggml_backend_buffer_i tb_buffer_iface = {
    /* .free_buffer     = */ tb_buffer_free,
    /* .get_base        = */ tb_buffer_get_base,
    /* .init_tensor     = */ tb_buffer_init_tensor,
    /* .memset_tensor   = */ tb_buffer_memset_tensor,
    /* .set_tensor      = */ tb_buffer_set_tensor,
    /* .get_tensor      = */ tb_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ tb_buffer_cpy_tensor,
    /* .clear           = */ tb_buffer_clear,
    /* .reset           = */ nullptr,
};

// -----------------------------------------------------------------------------
// Buffer type
// -----------------------------------------------------------------------------

static const char * tb_buft_get_name(ggml_backend_buffer_type_t /*buft*/) {
    return "TB-RDMA";
}

static ggml_backend_buffer_t tb_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * ep = static_cast<EndpointContext *>(buft->context);
    if (!ep || !ep->transport.connected()) {
        std::fprintf(stderr, "[tb-rdma] alloc_buffer: not connected\n");
        return nullptr;
    }

    MsgAllocBufferReq req = { size };
    MsgAllocBufferRsp rsp = {};
    if (ep->transport.request(CMD_ALLOC_BUFFER, req, CMD_ALLOC_BUFFER, rsp) != RequestStatus::Ok) {
        return nullptr;
    }
    if (rsp.remote_ptr == 0) return nullptr;

    auto * bc = new BufferContext;
    bc->ep          = get_or_create_endpoint(ep->endpoint);
    bc->remote_ptr  = rsp.remote_ptr;
    bc->remote_size = rsp.remote_size;
    return ggml_backend_buffer_init(buft, tb_buffer_iface, bc, rsp.remote_size);
}

static size_t tb_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    auto * ep = static_cast<EndpointContext *>(buft->context);
    if (!ep || !ep->transport.connected()) return 128;

    MsgGetAlignmentRsp rsp = {};
    if (ep->transport.request_no_payload(CMD_GET_ALIGNMENT, CMD_GET_ALIGNMENT, rsp) != RequestStatus::Ok) {
        return 128;
    }
    return rsp.alignment;
}

static size_t tb_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * ep = static_cast<EndpointContext *>(buft->context);
    if (!ep || !ep->transport.connected()) return SIZE_MAX;

    MsgGetMaxSizeRsp rsp = {};
    if (ep->transport.request_no_payload(CMD_GET_MAX_SIZE, CMD_GET_MAX_SIZE, rsp) != RequestStatus::Ok) {
        return SIZE_MAX;
    }
    return rsp.max_size;
}

static bool tb_buft_is_host(ggml_backend_buffer_type_t /*buft*/) {
    return false;
}

static ggml_backend_buffer_type_i tb_buft_iface = {
    /* .get_name       = */ tb_buft_get_name,
    /* .alloc_buffer   = */ tb_buft_alloc_buffer,
    /* .get_alignment  = */ tb_buft_get_alignment,
    /* .get_max_size   = */ tb_buft_get_max_size,
    /* .get_alloc_size = */ nullptr,
    /* .is_host        = */ tb_buft_is_host,
};

static ggml_backend_buffer_type_t buft_for_endpoint(std::shared_ptr<EndpointContext> ep) {
    std::lock_guard<std::mutex> lock(ep->mu);
    if (!ep->buft_inited) {
        ep->buft.iface   = tb_buft_iface;
        ep->buft.device  = nullptr;
        ep->buft.context = ep.get();
        ep->buft_inited  = true;
    }
    return &ep->buft;
}

// -----------------------------------------------------------------------------
// Backend interface
// -----------------------------------------------------------------------------

struct BackendContext {
    std::shared_ptr<EndpointContext> ep;
};

static const char * tb_backend_get_name(ggml_backend_t /*backend*/) {
    return "TB-RDMA";
}

static void tb_backend_free(ggml_backend_t backend) {
    auto * bc = static_cast<BackendContext *>(backend->context);
    delete bc;
    delete backend;
}

static enum ggml_status tb_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * bc = static_cast<BackendContext *>(backend->context);
    auto & tx = bc->ep->transport;

    // Iterative serialization (avoid stack overflow on deep graphs).
    std::vector<WireTensor> tensors;
    std::vector<uint64_t>   node_ids;
    std::unordered_set<ggml_tensor *> visited;
    std::vector<ggml_tensor *> stack;

    for (int i = 0; i < cgraph->n_nodes; i++) stack.push_back(cgraph->nodes[i]);
    while (!stack.empty()) {
        ggml_tensor * t = stack.back();
        stack.pop_back();
        if (!t || !visited.insert(t).second) continue;
        tensors.push_back(serialize_tensor(t));
        for (int i = 0; i < GGML_MAX_SRC; i++) if (t->src[i]) stack.push_back(t->src[i]);
        if (t->view_src) stack.push_back(t->view_src);
    }
    for (int i = 0; i < cgraph->n_nodes; i++) {
        node_ids.push_back(reinterpret_cast<uint64_t>(cgraph->nodes[i]));
    }

    MsgGraphComputeReq req = {};
    req.n_nodes   = static_cast<uint32_t>(node_ids.size());
    req.n_tensors = static_cast<uint32_t>(tensors.size());

    std::vector<Transport::Slice> parts = {
        { &req,            sizeof(req) },
        { node_ids.data(), node_ids.size() * sizeof(uint64_t) },
        { tensors.data(),  tensors.size()  * sizeof(WireTensor) },
    };
    std::vector<uint8_t> rsp_bytes;
    RequestStatus st = tx.request_var(CMD_GRAPH_COMPUTE, parts, CMD_GRAPH_COMPUTE,
                                       &rsp_bytes, sizeof(MsgGraphComputeRsp));
    if (st != RequestStatus::Ok || rsp_bytes.size() != sizeof(MsgGraphComputeRsp)) {
        return GGML_STATUS_FAILED;
    }
    MsgGraphComputeRsp rsp;
    std::memcpy(&rsp, rsp_bytes.data(), sizeof(rsp));
    return static_cast<ggml_status>(rsp.status);
}

static ggml_backend_i tb_backend_iface = {
    /* .get_name            = */ tb_backend_get_name,
    /* .free                = */ tb_backend_free,
    /* .set_tensor_async    = */ nullptr,
    /* .get_tensor_async    = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ nullptr,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ tb_backend_graph_compute,
    /* .event_record        = */ nullptr,
    /* .event_wait          = */ nullptr,
    /* .graph_optimize      = */ nullptr,
};

static ggml_guid_t tb_backend_guid() {
    static ggml_guid guid = {
        0x54, 0x42, 0x52, 0x44, 0x4D, 0x41, 0x00, 0x02,  // "TBRDMA\0\2"
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    };
    return &guid;
}

// -----------------------------------------------------------------------------
// Public C API
// -----------------------------------------------------------------------------

ggml_backend_t ggml_backend_tb_rdma_init(const char * endpoint_str, const char * rdma_device) {
    if (!endpoint_str) return nullptr;

    Endpoint ep_parsed;
    std::string err;
    if (!parse_endpoint(endpoint_str, ep_parsed, &err)) {
        std::fprintf(stderr, "[tb-rdma] bad endpoint '%s': %s\n", endpoint_str, err.c_str());
        return nullptr;
    }

    auto ep = get_or_create_endpoint(endpoint_str);

    {
        std::lock_guard<std::mutex> lock(ep->mu);
        if (!ep->transport.connected()) {
            ep->rdma_device = rdma_device ? rdma_device : "";
            if (!ep->transport.client_connect(ep_parsed, ep->rdma_device, &err)) {
                std::fprintf(stderr, "[tb-rdma] connect %s failed: %s\n", endpoint_str, err.c_str());
                return nullptr;
            }
        }
    }
    (void) buft_for_endpoint(ep);  // ensure buft is initialized

    auto * backend_ctx = new BackendContext;
    backend_ctx->ep    = ep;
    auto * backend  = new ggml_backend;
    backend->guid    = tb_backend_guid();
    backend->iface   = tb_backend_iface;
    backend->device  = nullptr;
    backend->context = backend_ctx;
    return backend;
}

bool ggml_backend_is_tb_rdma(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, tb_backend_guid());
}

ggml_backend_buffer_type_t ggml_backend_tb_rdma_buffer_type(const char * endpoint_str) {
    if (!endpoint_str) return nullptr;
    auto ep = get_or_create_endpoint(endpoint_str);
    return buft_for_endpoint(ep);
}

void ggml_backend_tb_rdma_get_device_memory(const char * endpoint_str, size_t * free, size_t * total) {
    if (free)  *free  = 0;
    if (total) *total = 0;
    if (!endpoint_str) return;

    auto ep = get_or_create_endpoint(endpoint_str);
    if (!ep->transport.connected()) {
        Endpoint parsed;
        if (!parse_endpoint(endpoint_str, parsed)) return;
        std::lock_guard<std::mutex> lock(ep->mu);
        if (!ep->transport.connected()) {
            if (!ep->transport.client_connect(parsed, ep->rdma_device)) return;
        }
    }
    MsgGetDeviceMemoryRsp rsp = {};
    if (ep->transport.request_no_payload(CMD_GET_DEVICE_MEMORY, CMD_GET_DEVICE_MEMORY, rsp) != RequestStatus::Ok) {
        return;
    }
    if (free)  *free  = rsp.free_mem;
    if (total) *total = rsp.total_mem;
}

void ggml_backend_tb_rdma_start_server(const char * endpoint_str,
                                       const char * rdma_device,
                                       ggml_backend_t backend) {
    if (!endpoint_str || !backend) return;
    run_server(endpoint_str, rdma_device ? rdma_device : "", backend);
}

// -----------------------------------------------------------------------------
// Device enumeration — RDMA-only inspection; not used until M2a.
// -----------------------------------------------------------------------------

bool ggml_tb_rdma_available(void) {
    return ibv().available && has_rdma_devices();
}

size_t ggml_tb_rdma_get_device_count(void) {
    const Ibv & v = ibv();
    if (!v.available) return 0;
    int n = 0;
    struct ibv_device ** list = v.get_device_list(&n);
    if (!list) return 0;
    const size_t count = n > 0 ? (size_t) n : 0;
    v.free_device_list(list);
    return count;
}

const char * ggml_tb_rdma_get_device_name(size_t index) {
    const Ibv & v = ibv();
    if (!v.available) return nullptr;
    int n = 0;
    struct ibv_device ** list = v.get_device_list(&n);
    if (!list || index >= (size_t) n) {
        if (list) v.free_device_list(list);
        return nullptr;
    }
    // Copy the name into a static thread_local buffer so the caller doesn't
    // hold a pointer into a freed device list.
    static thread_local char buf[128];
    const char * raw = v.get_device_name(list[index]);
    if (raw) {
        std::strncpy(buf, raw, sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';
    } else {
        buf[0] = '\0';
    }
    v.free_device_list(list);
    return buf[0] ? buf : nullptr;
}

// -----------------------------------------------------------------------------
// Backend registry
// -----------------------------------------------------------------------------

static const char * tb_reg_get_name(ggml_backend_reg_t /*reg*/) {
    return "TB-RDMA";
}

static size_t tb_reg_get_device_count(ggml_backend_reg_t /*reg*/) {
    // Devices are not pre-enumerated; the client picks endpoints at runtime.
    return 0;
}

static ggml_backend_dev_t tb_reg_get_device(ggml_backend_reg_t /*reg*/, size_t /*index*/) {
    return nullptr;
}

static void * tb_reg_get_proc_address(ggml_backend_reg_t /*reg*/, const char * name) {
    if (!name) return nullptr;
    if (std::strcmp(name, "ggml_backend_tb_rdma_init")             == 0) return reinterpret_cast<void *>(ggml_backend_tb_rdma_init);
    if (std::strcmp(name, "ggml_backend_is_tb_rdma")               == 0) return reinterpret_cast<void *>(ggml_backend_is_tb_rdma);
    if (std::strcmp(name, "ggml_backend_tb_rdma_buffer_type")      == 0) return reinterpret_cast<void *>(ggml_backend_tb_rdma_buffer_type);
    if (std::strcmp(name, "ggml_backend_tb_rdma_get_device_memory")== 0) return reinterpret_cast<void *>(ggml_backend_tb_rdma_get_device_memory);
    if (std::strcmp(name, "ggml_backend_tb_rdma_start_server")     == 0) return reinterpret_cast<void *>(ggml_backend_tb_rdma_start_server);
    if (std::strcmp(name, "ggml_tb_rdma_available")                == 0) return reinterpret_cast<void *>(ggml_tb_rdma_available);
    return nullptr;
}

static ggml_backend_reg_i tb_reg_iface = {
    /* .get_name         = */ tb_reg_get_name,
    /* .get_device_count = */ tb_reg_get_device_count,
    /* .get_device       = */ tb_reg_get_device,
    /* .get_proc_address = */ tb_reg_get_proc_address,
};

ggml_backend_reg_t ggml_backend_tb_rdma_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ tb_reg_iface,
        /* .context     = */ nullptr,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_tb_rdma_reg)

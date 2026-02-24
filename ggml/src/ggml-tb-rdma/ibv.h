#pragma once

// dlopen()-based loader for `librdma.dylib` (Apple Thunderbolt RDMA stack).
//
// We use Apple's SDK `<infiniband/verbs.h>` for struct/enum definitions
// (Apple ships a re-export tbd at /usr/lib/librdma.dylib that pulls in
// libibverbs internals). Hot-path verbs functions in the header are
// `static inline` and dispatch through `qp->context->ops.*` populated by
// the loaded library — they do not need dlsym. The non-inline functions
// (open_device, modify_qp, reg_mr, etc.) are resolved at runtime so the
// binary still loads on machines without librdma installed.
//
// `ibv().available` is false when librdma.dylib is missing. The backend
// uses this to drive the connection-time TCP fallback decision.

#include <infiniband/verbs.h>

#include <cstddef>
#include <cstdint>

namespace ggml_tb_rdma {

struct Ibv {
    bool         available = false;
    void *       handle    = nullptr;   // dlopen handle
    const char * lib_path  = nullptr;   // resolved path, for logging

    // Resolved verbs symbols (non-inline only).
    struct ibv_device ** (*get_device_list)(int * num_devices) = nullptr;
    void                 (*free_device_list)(struct ibv_device ** list) = nullptr;
    const char *         (*get_device_name)(struct ibv_device * device) = nullptr;
    struct ibv_context * (*open_device)(struct ibv_device * device) = nullptr;
    int                  (*close_device)(struct ibv_context * context) = nullptr;
    int                  (*query_device)(struct ibv_context * context, struct ibv_device_attr * attr) = nullptr;
    int                  (*query_port)(struct ibv_context * context, uint8_t port, struct ibv_port_attr * attr) = nullptr;
    int                  (*query_gid)(struct ibv_context * context, uint8_t port, int index, union ibv_gid * gid) = nullptr;
    struct ibv_pd *      (*alloc_pd)(struct ibv_context * context) = nullptr;
    int                  (*dealloc_pd)(struct ibv_pd * pd) = nullptr;
    struct ibv_mr *      (*reg_mr)(struct ibv_pd * pd, void * addr, size_t len, int access) = nullptr;
    int                  (*dereg_mr)(struct ibv_mr * mr) = nullptr;
    struct ibv_cq *      (*create_cq)(struct ibv_context * context, int cqe, void * cq_context,
                                      struct ibv_comp_channel * channel, int vector) = nullptr;
    int                  (*destroy_cq)(struct ibv_cq * cq) = nullptr;
    struct ibv_qp *      (*create_qp)(struct ibv_pd * pd, struct ibv_qp_init_attr * attr) = nullptr;
    int                  (*modify_qp)(struct ibv_qp * qp, struct ibv_qp_attr * attr, int attr_mask) = nullptr;
    int                  (*destroy_qp)(struct ibv_qp * qp) = nullptr;
};

// Lazily initialized singleton. Repeated calls return the same instance.
// Thread-safe.
const Ibv & ibv();

// True when `ibv().available` AND at least one device exists.
bool has_rdma_devices();

} // namespace ggml_tb_rdma

#include "ibv.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace ggml_tb_rdma {

namespace {

// Search order for the verbs library. Apple's official location is
// /usr/lib/librdma.dylib (re-exports libibverbs internals); we also accept
// the bare basename (search dyld paths) so installations under
// /usr/local/lib still work.
const char * const kLibPaths[] = {
    "librdma.dylib",
    "/usr/lib/librdma.dylib",
    "libibverbs.dylib",
    "/usr/lib/libibverbs.dylib",
    nullptr,
};

Ibv g_ibv;
std::once_flag g_once;

template <typename Fn>
bool load_sym(void * h, const char * name, Fn & out) {
    void * p = dlsym(h, name);
    if (!p) return false;
    out = reinterpret_cast<Fn>(p);
    return true;
}

void load_once() {
    void * h = nullptr;
    const char * found = nullptr;
    for (const char * const * pp = kLibPaths; *pp; ++pp) {
        h = dlopen(*pp, RTLD_NOW | RTLD_LOCAL);
        if (h) { found = *pp; break; }
    }
    if (!h) {
        // Quiet by default — the backend will log the fallback decision once
        // it has the full context (e.g. "RDMA setup failed: <reason>...").
        return;
    }

    Ibv & v = g_ibv;
    v.handle   = h;
    v.lib_path = found;

    bool ok = true;
    ok &= load_sym(h, "ibv_get_device_list", v.get_device_list);
    ok &= load_sym(h, "ibv_free_device_list", v.free_device_list);
    ok &= load_sym(h, "ibv_get_device_name", v.get_device_name);
    ok &= load_sym(h, "ibv_open_device", v.open_device);
    ok &= load_sym(h, "ibv_close_device", v.close_device);
    ok &= load_sym(h, "ibv_query_device", v.query_device);
    ok &= load_sym(h, "ibv_query_port", v.query_port);
    ok &= load_sym(h, "ibv_query_gid", v.query_gid);
    ok &= load_sym(h, "ibv_alloc_pd", v.alloc_pd);
    ok &= load_sym(h, "ibv_dealloc_pd", v.dealloc_pd);
    ok &= load_sym(h, "ibv_reg_mr", v.reg_mr);
    ok &= load_sym(h, "ibv_dereg_mr", v.dereg_mr);
    ok &= load_sym(h, "ibv_create_cq", v.create_cq);
    ok &= load_sym(h, "ibv_destroy_cq", v.destroy_cq);
    ok &= load_sym(h, "ibv_create_qp", v.create_qp);
    ok &= load_sym(h, "ibv_modify_qp", v.modify_qp);
    ok &= load_sym(h, "ibv_destroy_qp", v.destroy_qp);

    if (!ok) {
        // Partial resolution → treat as unavailable. Don't dlclose: keep the
        // handle live for the lifetime of the process (verbs libs are not
        // safe to unload while objects exist anyway).
        v = Ibv{};
        v.handle   = h;
        v.lib_path = found;
        return;
    }

    v.available = true;
}

} // namespace

const Ibv & ibv() {
    std::call_once(g_once, load_once);
    return g_ibv;
}

bool has_rdma_devices() {
    const Ibv & v = ibv();
    if (!v.available) return false;
    int n = 0;
    struct ibv_device ** list = v.get_device_list(&n);
    if (!list) return false;
    const bool any = (n > 0);
    v.free_device_list(list);
    return any;
}

} // namespace ggml_tb_rdma

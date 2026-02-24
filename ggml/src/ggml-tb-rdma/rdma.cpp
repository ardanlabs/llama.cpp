#include "rdma.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace ggml_tb_rdma {

// -----------------------------------------------------------------------------
// env-var helpers (also used by transport.cpp via the header)
// -----------------------------------------------------------------------------

bool env_truthy(const char * name) {
    const char * v = std::getenv(name);
    if (!v || !*v) return false;
    return v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y';
}

int env_int(const char * name, int defv) {
    const char * v = std::getenv(name);
    if (!v || !*v) return defv;
    char * end = nullptr;
    long n = std::strtol(v, &end, 10);
    if (end == v) return defv;
    return (int) n;
}

namespace {

// -----------------------------------------------------------------------------
// Logging
// -----------------------------------------------------------------------------

enum LogLevel { LOG_SILENT = 0, LOG_ERROR = 1, LOG_INFO = 2, LOG_DEBUG = 3 };

LogLevel log_level() {
    static LogLevel cached = []() {
        const char * v = std::getenv("GGML_TB_RDMA_LOG_LEVEL");
        if (!v || !*v) return LOG_ERROR;
        if (!std::strcmp(v, "silent")) return LOG_SILENT;
        if (!std::strcmp(v, "error"))  return LOG_ERROR;
        if (!std::strcmp(v, "info"))   return LOG_INFO;
        if (!std::strcmp(v, "debug"))  return LOG_DEBUG;
        return LOG_ERROR;
    }();
    return cached;
}

#define TBLOG(lvl, ...) do { if (log_level() >= (lvl)) std::fprintf(stderr, __VA_ARGS__); } while(0)

int64_t now_ms() {
    auto t = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(t).count();
}

uint64_t gen_nonce() {
    auto t = std::chrono::steady_clock::now().time_since_epoch();
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(t).count();
}

// Path MTU → bytes (verbs encodes as enum).
uint32_t mtu_to_bytes(enum ibv_mtu mtu) {
    switch (mtu) {
        case IBV_MTU_256:  return 256;
        case IBV_MTU_512:  return 512;
        case IBV_MTU_1024: return 1024;
        case IBV_MTU_2048: return 2048;
        case IBV_MTU_4096: return 4096;
    }
    return 1024;
}

// Minimum of two MTU enums (returns the smaller bytes-wise).
enum ibv_mtu mtu_min(enum ibv_mtu a, enum ibv_mtu b) {
    return mtu_to_bytes(a) < mtu_to_bytes(b) ? a : b;
}

// Format a 16-byte GID for diagnostics.
std::string gid_to_string(const union ibv_gid & g) {
    char buf[64];
    std::snprintf(buf, sizeof(buf),
                  "%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x",
                  g.raw[0],  g.raw[1],  g.raw[2],  g.raw[3],
                  g.raw[4],  g.raw[5],  g.raw[6],  g.raw[7],
                  g.raw[8],  g.raw[9],  g.raw[10], g.raw[11],
                  g.raw[12], g.raw[13], g.raw[14], g.raw[15]);
    return buf;
}

// Convert an IP string into the GID bytes we'd expect for RoCEv2 (IPv4
// mapped) so we can match against the GID table.
bool ip_to_gid(const std::string & ip, union ibv_gid & out) {
    std::memset(&out, 0, sizeof(out));
    in6_addr v6;
    if (inet_pton(AF_INET6, ip.c_str(), &v6) == 1) {
        std::memcpy(out.raw, &v6, 16);
        return true;
    }
    in_addr v4;
    if (inet_pton(AF_INET, ip.c_str(), &v4) == 1) {
        // IPv4-mapped IPv6
        std::memset(out.raw, 0, 10);
        out.raw[10] = 0xff;
        out.raw[11] = 0xff;
        std::memcpy(out.raw + 12, &v4, 4);
        return true;
    }
    return false;
}

// Discover the local IP this TCP socket used.
std::string local_ip_of(int fd) {
    sockaddr_storage ss = {};
    socklen_t        sl = sizeof(ss);
    if (::getsockname(fd, (sockaddr *) &ss, &sl) != 0) return "";
    char buf[INET6_ADDRSTRLEN] = {0};
    if (ss.ss_family == AF_INET) {
        inet_ntop(AF_INET, &reinterpret_cast<sockaddr_in &>(ss).sin_addr, buf, sizeof(buf));
    } else if (ss.ss_family == AF_INET6) {
        inet_ntop(AF_INET6, &reinterpret_cast<sockaddr_in6 &>(ss).sin6_addr, buf, sizeof(buf));
    }
    return buf;
}

} // namespace

// -----------------------------------------------------------------------------
// RdmaConnection: destruction
// -----------------------------------------------------------------------------

RdmaConnection::~RdmaConnection() { close(); }

void RdmaConnection::close() {
    std::lock_guard<std::mutex> lock(mu_);
    drain_and_destroy();
}

void RdmaConnection::drain_and_destroy() {
    const Ibv & v = ibv();
    if (qp_) {
        // Clean QP teardown (fix for bug I).
        struct ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_ERR;
        if (v.modify_qp) (void) v.modify_qp(qp_, &a, IBV_QP_STATE);
        if (v.destroy_qp) v.destroy_qp(qp_);
        qp_ = nullptr;
    }
    if (cq_ && v.destroy_cq) { v.destroy_cq(cq_); cq_ = nullptr; }
    if (send_mr_ && v.dereg_mr) { v.dereg_mr(send_mr_); send_mr_ = nullptr; }
    if (recv_mr_ && v.dereg_mr) { v.dereg_mr(recv_mr_); recv_mr_ = nullptr; }
    if (pd_ && v.dealloc_pd) { v.dealloc_pd(pd_); pd_ = nullptr; }
    if (ctx_ && v.close_device) { v.close_device(ctx_); ctx_ = nullptr; }
    if (send_buf_) { std::free(send_buf_); send_buf_ = nullptr; send_buf_sz_ = 0; }
    if (recv_buf_) { std::free(recv_buf_); recv_buf_ = nullptr; recv_buf_sz_ = 0; }
    alive_ = false;
}

// -----------------------------------------------------------------------------
// Bring-up: open_device, GID selection
// -----------------------------------------------------------------------------

bool RdmaConnection::open_device(std::string * err) {
    const Ibv & v = ibv();
    if (!v.available) {
        if (err) *err = "librdma.dylib not available";
        return false;
    }

    int n = 0;
    struct ibv_device ** list = v.get_device_list(&n);
    if (!list || n <= 0) {
        if (err) *err = "no RDMA devices found";
        if (list) v.free_device_list(list);
        return false;
    }

    const char * want = std::getenv("GGML_TB_RDMA_DEVICE");
    struct ibv_device * dev = nullptr;
    for (int i = 0; i < n; i++) {
        const char * name = v.get_device_name(list[i]);
        if (want && std::strcmp(want, name) == 0) { dev = list[i]; break; }
    }
    if (!dev) dev = list[0];

    ctx_ = v.open_device(dev);
    v.free_device_list(list);
    if (!ctx_) {
        if (err) *err = "ibv_open_device failed";
        return false;
    }

    pd_ = v.alloc_pd(ctx_);
    if (!pd_) {
        if (err) *err = "ibv_alloc_pd failed";
        return false;
    }

    // Pick port 1 by default; query to discover MTU and GID table size.
    struct ibv_port_attr pa = {};
    if (v.query_port(ctx_, port_num_, &pa) != 0) {
        if (err) *err = "ibv_query_port failed";
        return false;
    }
    path_mtu_  = (uint8_t) pa.active_mtu;
    frame_size_ = mtu_to_bytes(pa.active_mtu);
    local_lid_  = pa.lid;

    // GID selection.
    int gid_override = env_int("GGML_TB_RDMA_GID_INDEX", -1);
    if (gid_override >= 0 && gid_override < pa.gid_tbl_len) {
        gid_index_ = (uint8_t) gid_override;
    } else {
        // Walk the table; prefer an entry whose GID equals our local TCP IP.
        union ibv_gid want_gid = {};
        bool have_want = !local_ip_.empty() && ip_to_gid(local_ip_, want_gid);
        int picked = -1;
        for (int i = 0; i < pa.gid_tbl_len; i++) {
            union ibv_gid g = {};
            if (v.query_gid(ctx_, port_num_, i, &g) != 0) continue;
            // Skip zero GIDs.
            bool zero = true;
            for (int k = 0; k < 16; k++) if (g.raw[k]) { zero = false; break; }
            if (zero) continue;
            if (have_want && std::memcmp(g.raw, want_gid.raw, 16) == 0) {
                picked = i;
                break;
            }
            if (picked < 0) picked = i;  // first non-zero fallback
        }
        if (picked < 0) {
            if (err) *err = "no usable GID";
            return false;
        }
        if (have_want) {
            union ibv_gid g = {};
            v.query_gid(ctx_, port_num_, picked, &g);
            if (std::memcmp(g.raw, want_gid.raw, 16) != 0) {
                TBLOG(LOG_INFO,
                      "[tb-rdma] GID exact match for local IP %s not found; using index %d\n",
                      local_ip_.c_str(), picked);
            }
        }
        gid_index_ = (uint8_t) picked;
    }
    if (v.query_gid(ctx_, port_num_, gid_index_, &local_gid_) != 0) {
        if (err) *err = "ibv_query_gid failed";
        return false;
    }
    TBLOG(LOG_INFO, "[tb-rdma] using GID index %u: %s\n", gid_index_, gid_to_string(local_gid_).c_str());

    // Query device caps to clamp R/W.
    struct ibv_device_attr da = {};
    v.query_device(ctx_, &da);
    uint32_t cap_qp_wr = (uint32_t) da.max_qp_wr;
    if (cap_qp_wr == 0) cap_qp_wr = 256;

    recv_credits_ = std::min<uint32_t>(TB_RDMA_DEFAULT_RECV_CREDITS, cap_qp_wr);
    send_window_  = std::min<uint32_t>(TB_RDMA_DEFAULT_SEND_WINDOW, cap_qp_wr);
    if (recv_credits_ == 0) recv_credits_ = 16;
    if (send_window_  == 0) send_window_  = 4;

    local_psn_ = (uint32_t) (now_ms() & 0xFFFFFF);
    return true;
}

bool RdmaConnection::register_mrs(std::string * err) {
    const Ibv & v = ibv();

    // TN3205: register page-aligned memory. Round the total allocation up to
    // a multiple of the page size as well, so the IOMMU mapping never spans
    // an unaligned tail.
    const long ps = sysconf(_SC_PAGESIZE);
    const size_t page = ps > 0 ? (size_t) ps : 16384;  // Apple Silicon default
    auto round_up = [page](size_t n) { return (n + page - 1) & ~(page - 1); };

    send_buf_sz_ = round_up((size_t) send_window_  * frame_size_);
    recv_buf_sz_ = round_up((size_t) recv_credits_ * frame_size_);

    void * sp = nullptr;
    void * rp = nullptr;
    if (posix_memalign(&sp, page, send_buf_sz_) != 0 ||
        posix_memalign(&rp, page, recv_buf_sz_) != 0) {
        if (sp) std::free(sp);
        if (rp) std::free(rp);
        if (err) *err = "posix_memalign failed";
        return false;
    }
    std::memset(sp, 0, send_buf_sz_);
    std::memset(rp, 0, recv_buf_sz_);
    send_buf_ = (uint8_t *) sp;
    recv_buf_ = (uint8_t *) rp;

    const int access = IBV_ACCESS_LOCAL_WRITE;
    send_mr_ = v.reg_mr(pd_, send_buf_, send_buf_sz_, access);
    recv_mr_ = v.reg_mr(pd_, recv_buf_, recv_buf_sz_, access);
    if (!send_mr_ || !recv_mr_) {
        if (err) *err = "ibv_reg_mr failed";
        return false;
    }
    send_slots_.assign(send_window_, SendSlot{});
    return true;
}

bool RdmaConnection::create_qp(std::string * err) {
    const Ibv & v = ibv();
    cq_ = v.create_cq(ctx_, (int) (send_window_ + recv_credits_) * 2, nullptr, nullptr, 0);
    if (!cq_) { if (err) *err = "ibv_create_cq failed"; return false; }

    struct ibv_qp_init_attr ia = {};
    ia.send_cq = cq_;
    ia.recv_cq = cq_;
    ia.qp_type = IBV_QPT_UC;             // Apple Thunderbolt only allows UC
    ia.cap.max_send_wr  = send_window_;
    ia.cap.max_recv_wr  = recv_credits_;
    ia.cap.max_send_sge = 1;
    ia.cap.max_recv_sge = 1;
    ia.sq_sig_all       = 1;             // we want a CQE for every send

    qp_ = v.create_qp(pd_, &ia);
    if (!qp_) { if (err) *err = "ibv_create_qp failed"; return false; }
    return true;
}

bool RdmaConnection::transition_to_init(std::string * err) {
    const Ibv & v = ibv();
    struct ibv_qp_attr a = {};
    a.qp_state        = IBV_QPS_INIT;
    a.pkey_index      = 0;
    a.port_num        = port_num_;
    a.qp_access_flags = 0;               // UC: no remote write/read
    int mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    if (v.modify_qp(qp_, &a, mask) != 0) {
        if (err) *err = "modify_qp INIT failed";
        return false;
    }
    return true;
}

bool RdmaConnection::transition_to_rtr(const MsgQPInfo & peer, std::string * err) {
    const Ibv & v = ibv();
    struct ibv_qp_attr a = {};
    a.qp_state           = IBV_QPS_RTR;
    a.path_mtu           = mtu_min((enum ibv_mtu) path_mtu_, (enum ibv_mtu) peer.path_mtu);
    a.dest_qp_num        = peer.qp_num;
    a.rq_psn             = peer.psn;
    a.ah_attr.is_global  = 1;
    a.ah_attr.dlid       = peer.lid;       // TN3205: peer port LID from ibv_query_port
    a.ah_attr.sl         = 0;
    a.ah_attr.src_path_bits = 0;
    a.ah_attr.port_num   = port_num_;
    std::memcpy(a.ah_attr.grh.dgid.raw, peer.gid.raw, 16);
    a.ah_attr.grh.sgid_index = gid_index_;
    a.ah_attr.grh.hop_limit  = 1;          // TN3205: single-hop Thunderbolt fabric
    a.ah_attr.grh.traffic_class = 0;
    int mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
               IBV_QP_DEST_QPN | IBV_QP_RQ_PSN;
    if (v.modify_qp(qp_, &a, mask) != 0) {
        if (err) *err = "modify_qp RTR failed";
        return false;
    }
    // Lock down the effective frame size to the negotiated MTU.
    frame_size_ = std::min(frame_size_, mtu_to_bytes(a.path_mtu));
    return true;
}

bool RdmaConnection::transition_to_rts(std::string * err) {
    const Ibv & v = ibv();
    struct ibv_qp_attr a = {};
    a.qp_state = IBV_QPS_RTS;
    a.sq_psn   = local_psn_;
    int mask = IBV_QP_STATE | IBV_QP_SQ_PSN;
    if (v.modify_qp(qp_, &a, mask) != 0) {
        if (err) *err = "modify_qp RTS failed";
        return false;
    }
    return true;
}

bool RdmaConnection::post_initial_recvs(std::string * err) {
    for (uint32_t i = 0; i < recv_credits_; i++) {
        if (!post_recv_slot(i)) {
            if (err) *err = "ibv_post_recv failed";
            return false;
        }
    }
    return true;
}

bool RdmaConnection::post_recv_slot(uint32_t idx) {
    if (!qp_) return false;
    struct ibv_sge sge = {};
    sge.addr   = (uintptr_t) (recv_buf_ + (size_t) idx * frame_size_);
    sge.length = frame_size_;
    sge.lkey   = recv_mr_->lkey;
    struct ibv_recv_wr wr = {};
    wr.wr_id   = (uint64_t) idx | (1ULL << 63);  // mark as recv
    wr.sg_list = &sge;
    wr.num_sge = 1;
    struct ibv_recv_wr * bad = nullptr;
    return ibv_post_recv(qp_, &wr, &bad) == 0;
}

bool RdmaConnection::post_send_slot(uint32_t idx, uint16_t payload_len) {
    if (!qp_) return false;
#ifdef GGML_TB_RDMA_FAULT_INJECT
    out_frame_counter_++;
    if (drop_every_ && (out_frame_counter_ % drop_every_) == 0) {
        // Pretend we sent it — Apple UC drops silently, so peer just won't
        // see it. We still mark the slot in-flight for retransmit logic.
        stats_.frames_sent += 1;
        stats_.bytes_sent  += payload_len;
        return true;
    }
#endif
    struct ibv_sge sge = {};
    sge.addr   = (uintptr_t) (send_buf_ + (size_t) idx * frame_size_);
    sge.length = payload_len;
    sge.lkey   = send_mr_->lkey;
    struct ibv_send_wr wr = {};
    wr.wr_id      = (uint64_t) idx;  // recv flag = bit 63 = 0
    wr.sg_list    = &sge;
    wr.num_sge    = 1;
    wr.opcode     = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;
    struct ibv_send_wr * bad = nullptr;
    int rc = ibv_post_send(qp_, &wr, &bad);
    if (rc != 0) {
        stats_.qp_cq_errors++;
        return false;
    }
    stats_.frames_sent += 1;
    stats_.bytes_sent  += payload_len;
    return true;
}

// -----------------------------------------------------------------------------
// Top-level bring-up
// -----------------------------------------------------------------------------

bool RdmaConnection::bring_up(int tcp_fd, std::string * err) {
    std::lock_guard<std::mutex> lock(mu_);
    if (alive_) { if (err) *err = "already alive"; return false; }
    local_ip_ = local_ip_of(tcp_fd);
    if (!open_device(err))     { drain_and_destroy(); return false; }
    if (!create_qp(err))       { drain_and_destroy(); return false; }
    if (!register_mrs(err))    { drain_and_destroy(); return false; }
    if (!transition_to_init(err)) { drain_and_destroy(); return false; }
    if (!post_initial_recvs(err)) { drain_and_destroy(); return false; }
    return true;
}

MsgQPInfo RdmaConnection::local_qp_info() const {
    MsgQPInfo m = {};
    m.qp_num       = qp_ ? qp_->qp_num : 0;
    m.recv_credits = recv_credits_;
    m.frame_size   = frame_size_;
    m.port_num     = port_num_;
    m.gid_index    = gid_index_;
    m.path_mtu     = path_mtu_;
    m.psn          = local_psn_;
    std::memcpy(m.gid.raw, local_gid_.raw, 16);
    m.lid          = local_lid_;
    return m;
}

bool RdmaConnection::finalize(const MsgQPInfo & peer, std::string * err) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!transition_to_rtr(peer, err)) { drain_and_destroy(); return false; }
    if (!transition_to_rts(err))       { drain_and_destroy(); return false; }
    peer_credits_     = peer.recv_credits;
    peer_credits_max_ = peer.recv_credits;
    alive_ = true;
    return true;
}

// -----------------------------------------------------------------------------
// Readiness ping — exchange a nonce over RDMA.
// -----------------------------------------------------------------------------

bool RdmaConnection::readiness_ping(bool is_server, std::string * err, int timeout_ms) {
    // Client sends first, server echoes; both verify byte-exact.
    const uint64_t my_nonce = gen_nonce();
    if (!is_server) {
        if (send_ctrl(&my_nonce, sizeof(my_nonce), timeout_ms) != RdmaResult::Ok) {
            if (err) *err = "readiness send failed";
            return false;
        }
        std::vector<uint8_t> echo;
        if (recv_ctrl(&echo, timeout_ms) != RdmaResult::Ok || echo.size() != sizeof(uint64_t)) {
            if (err) *err = "readiness recv failed";
            return false;
        }
        uint64_t back = 0; std::memcpy(&back, echo.data(), sizeof(back));
        if (back != my_nonce) {
            if (err) *err = "readiness nonce mismatch";
            return false;
        }
    } else {
        std::vector<uint8_t> req;
        if (recv_ctrl(&req, timeout_ms) != RdmaResult::Ok || req.size() != sizeof(uint64_t)) {
            if (err) *err = "readiness recv failed";
            return false;
        }
        if (send_ctrl(req.data(), req.size(), timeout_ms) != RdmaResult::Ok) {
            if (err) *err = "readiness send failed";
            return false;
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
// Progress engine
// -----------------------------------------------------------------------------

void RdmaConnection::poll_completions() {
    struct ibv_wc wc[16];
    for (;;) {
        int n = ibv_poll_cq(cq_, 16, wc);
        if (n <= 0) return;
        for (int i = 0; i < n; i++) {
            const bool is_recv = (wc[i].wr_id >> 63) & 1ULL;
            const uint32_t idx = (uint32_t) (wc[i].wr_id & ~(1ULL << 63));
            if (wc[i].status != IBV_WC_SUCCESS) {
                stats_.qp_cq_errors++;
                TBLOG(LOG_ERROR, "[tb-rdma] CQE error status=%d\n", wc[i].status);
                alive_ = false;
                continue;
            }
            if (is_recv) {
                stats_.frames_received += 1;
                stats_.bytes_received  += wc[i].byte_len;
                if (wc[i].byte_len < sizeof(RdmaFrame)) {
                    TBLOG(LOG_ERROR, "[tb-rdma] short recv: %u bytes\n", wc[i].byte_len);
                    alive_ = false;
                    continue;
                }
                const uint8_t * base = recv_buf_ + (size_t) idx * frame_size_;
                RdmaFrame h = {};
                std::memcpy(&h, base, sizeof(h));
                on_recv_frame(h, base + sizeof(RdmaFrame), idx);
                // Repost the slot for further receives.
                (void) post_recv_slot(idx);
            } else {
                if (idx < send_slots_.size()) {
                    send_slots_[idx].in_flight = false;
                    // We don't mark "acked" yet — that's by peer ACK, not by
                    // local completion. But the slot may be reused if it has
                    // already been peer-acked (see on_recv_frame ack logic).
                }
            }
        }
    }
}

void RdmaConnection::on_recv_frame(const RdmaFrame & h, const uint8_t * payload, uint32_t /*idx*/) {
    // Apply peer ack first (frees send slots regardless of frame kind).
    if (h.ack > last_acked_seq_) {
        // Each newly cumulatively-acked seq returns one peer-side recv credit:
        // every DATA/CTRL frame we sent consumed exactly one peer credit on
        // post, and the peer reposts its recv slot after consuming the frame.
        // FLOW frames carry seq == 0 and do not consume peer credits, so the
        // 1:1 mapping between advancing seqs and returned credits holds.
        const uint32_t newly_acked = h.ack - last_acked_seq_;
        peer_credits_ += newly_acked;
        if (peer_credits_max_ && peer_credits_ > peer_credits_max_) {
            peer_credits_ = peer_credits_max_;
        }
        for (auto & s : send_slots_) {
            if (s.in_flight == false && s.payload_len > 0 && s.seq <= h.ack) {
                s.payload_len = 0;
                s.acked = true;
                if (in_flight_count_ > 0) in_flight_count_--;
            }
        }
        last_acked_seq_ = h.ack;
    }

    if (h.kind == RDMA_KIND_FLOW) {
        // ACK-only, nothing more to do.
        return;
    }

    if (h.seq == 0) {
        // Should never happen; first valid seq is 1.
        return;
    }

    if (h.seq < expected_recv_seq_) {
        // Duplicate — drop, but re-emit our ack so peer can advance.
        ack_pending_ = true;
        return;
    }
    if (h.seq > expected_recv_seq_) {
        // Out-of-order — drop and ACK current (go-back-N).
        ack_pending_ = true;
        return;
    }
    // In-order. Consume.
    expected_recv_seq_++;
    ack_pending_ = true;

    if (h.kind == RDMA_KIND_CTRL) {
        auto & st = ctrl_pending_[h.msg_id];
        if (h.flags & RDMA_FLAG_FIRST) {
            st.total = h.payload_offset_or_total;
            st.bytes.clear();
            st.bytes.reserve((size_t) std::min<uint64_t>(st.total, MAX_CONTROL_FRAME_BYTES));
        }
        if (st.bytes.size() + h.len > MAX_CONTROL_FRAME_BYTES) {
            TBLOG(LOG_ERROR, "[tb-rdma] CTRL too big\n");
            alive_ = false;
            return;
        }
        st.bytes.insert(st.bytes.end(), payload, payload + h.len);
        if (h.flags & RDMA_FLAG_LAST) {
            ctrl_ready_.push_back(std::move(st.bytes));
            ctrl_pending_.erase(h.msg_id);
        }
    } else if (h.kind == RDMA_KIND_DATA) {
        if (h.flags & RDMA_FLAG_FIRST) {
            data_recv_.active = true;
            data_recv_.total  = h.payload_offset_or_total;
            data_recv_.copied = 0;
            data_recv_.msg_id = h.msg_id;
            // dest was set by recv_data() before progress started.
            assert(data_recv_.dest && "DATA recv with no registered destination");
        }
        if (!data_recv_.active || h.msg_id != data_recv_.msg_id) {
            TBLOG(LOG_ERROR, "[tb-rdma] unexpected DATA frame\n");
            alive_ = false;
            return;
        }
        uint64_t dst_off = (h.flags & RDMA_FLAG_FIRST) ? 0 : h.payload_offset_or_total;
        if (dst_off + h.len > data_recv_.total) {
            TBLOG(LOG_ERROR, "[tb-rdma] DATA frame overruns total\n");
            alive_ = false;
            return;
        }
        std::memcpy((uint8_t *) data_recv_.dest + dst_off, payload, h.len);
        data_recv_.copied += h.len;
        if (h.flags & RDMA_FLAG_LAST) {
            data_recv_pending_complete_ = true;
        }
    }
}

void RdmaConnection::maybe_emit_ack() {
    if (!alive_ || !ack_pending_) return;
    const int64_t t = now_ms();
    if (t - last_ack_emit_ms_ < TB_RDMA_ACK_DELAY_MS) return;
    // Send a FLOW frame with cumulative ack = expected_recv_seq_ - 1.
    // Use a fresh send slot — but FLOW doesn't consume peer credits in our
    // simple model (FLOW is bookkeeping, not data).
    uint32_t idx = next_send_slot_;
    for (uint32_t i = 0; i < send_window_; i++) {
        uint32_t k = (next_send_slot_ + i) % send_window_;
        if (!send_slots_[k].in_flight && send_slots_[k].payload_len == 0) { idx = k; break; }
    }
    if (send_slots_[idx].in_flight || send_slots_[idx].payload_len != 0) return;  // try later

    uint8_t * slot = send_buf_ + (size_t) idx * frame_size_;
    RdmaFrame h = {};
    h.msg_id = 0;
    h.seq    = 0;   // FLOW frames carry no payload seq
    h.ack    = expected_recv_seq_ - 1;
    h.kind   = RDMA_KIND_FLOW;
    h.flags  = 0;
    h.len    = 0;
    h.payload_offset_or_total = 0;
    std::memcpy(slot, &h, sizeof(h));
    send_slots_[idx].in_flight   = true;
    send_slots_[idx].payload_len = sizeof(RdmaFrame);
    send_slots_[idx].seq         = 0;
    if (!post_send_slot(idx, sizeof(RdmaFrame))) {
        send_slots_[idx].in_flight = false;
        send_slots_[idx].payload_len = 0;
        return;
    }
    stats_.flow_only_frames++;
    last_ack_emit_ms_ = t;
    ack_pending_ = false;
}

void RdmaConnection::maybe_retransmit() {
    const int64_t t = now_ms();
    for (auto & s : send_slots_) {
        if (s.payload_len > 0 && !s.in_flight && !s.acked) {
            if (t - s.last_post_ms < TB_RDMA_RETRANSMIT_TIMEOUT_MS) continue;
            if (s.retransmits >= TB_RDMA_MAX_RETRANSMITS) {
                alive_ = false;
                return;
            }
            // The slot still holds the prepared frame bytes. Re-post.
            uint32_t idx = (uint32_t) (&s - &send_slots_[0]);
            s.in_flight = true;
            s.retransmits++;
            s.last_post_ms = t;
            (void) post_send_slot(idx, s.payload_len);
            stats_.retransmits++;
        }
    }
}

RdmaResult RdmaConnection::progress_until(const Predicate & p, int timeout_ms) {
    const int64_t deadline = now_ms() + timeout_ms;
    for (;;) {
        poll_completions();
        if (!alive_) return RdmaResult::Dead;
        maybe_emit_ack();
        maybe_retransmit();
        if (!alive_) return RdmaResult::Dead;
        if (p()) return RdmaResult::Ok;
        if (now_ms() >= deadline) return RdmaResult::Timeout;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

// -----------------------------------------------------------------------------
// Public message API
// -----------------------------------------------------------------------------

namespace {
size_t total_size(const std::vector<RdmaConnection::Slice> & parts) {
    size_t t = 0; for (auto & p : parts) t += p.size; return t;
}
} // namespace

RdmaResult RdmaConnection::send_ctrl(const void * data, size_t len, int timeout_ms) {
    Slice s{ data, len };
    return send_ctrl_parts({ s }, timeout_ms);
}

RdmaResult RdmaConnection::send_ctrl_parts(const std::vector<Slice> & parts, int timeout_ms) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!alive_) return RdmaResult::Closed;
    size_t total = total_size(parts);
    if (total > MAX_CONTROL_FRAME_BYTES) return RdmaResult::ProtocolError;

    const uint32_t msg_id = next_msg_id_++;
    const uint32_t per    = payload_per_frame();
    const uint32_t n_frames = per ? (uint32_t) ((total + per - 1) / per) : 1;

    // Linearize parts for easy chunking (simple, fine for ≤ 1 MiB CTRL).
    std::vector<uint8_t> buf;
    buf.reserve(total);
    for (auto & p : parts) {
        if (p.size) buf.insert(buf.end(),
                                (const uint8_t *) p.data,
                                (const uint8_t *) p.data + p.size);
    }

    size_t off = 0;
    for (uint32_t i = 0; i < n_frames; i++) {
        // Wait for free slot + peer credit.
        RdmaResult r = progress_until([&]{
            for (auto & s : send_slots_) {
                if (!s.in_flight && s.payload_len == 0) return peer_credits_ > 0;
            }
            return false;
        }, timeout_ms);
        if (r != RdmaResult::Ok) {
            if (r == RdmaResult::Timeout) stats_.credit_stalls++;
            return r;
        }
        uint32_t idx = 0;
        for (; idx < send_window_; idx++) if (!send_slots_[idx].in_flight && send_slots_[idx].payload_len == 0) break;

        const uint16_t this_len = (uint16_t) std::min<size_t>(per, total - off);
        uint8_t * slot = send_buf_ + (size_t) idx * frame_size_;
        RdmaFrame h = {};
        h.msg_id = msg_id;
        h.seq    = next_send_seq_++;
        h.ack    = expected_recv_seq_ - 1;
        h.kind   = RDMA_KIND_CTRL;
        h.flags  = (i == 0 ? RDMA_FLAG_FIRST : 0) | (i == n_frames - 1 ? RDMA_FLAG_LAST : 0);
        h.len    = this_len;
        h.payload_offset_or_total = (i == 0) ? (uint64_t) total : 0;
        std::memcpy(slot, &h, sizeof(h));
        if (this_len) std::memcpy(slot + sizeof(RdmaFrame), buf.data() + off, this_len);

        send_slots_[idx].in_flight   = true;
        send_slots_[idx].acked       = false;
        send_slots_[idx].payload_len = sizeof(RdmaFrame) + this_len;
        send_slots_[idx].seq         = h.seq;
        send_slots_[idx].retransmits = 0;
        send_slots_[idx].last_post_ms = now_ms();
        if (!post_send_slot(idx, (uint16_t) send_slots_[idx].payload_len)) {
            send_slots_[idx] = SendSlot{};
            return RdmaResult::Dead;
        }
        in_flight_count_++;
        if (peer_credits_) peer_credits_--;
        off += this_len;
    }
    // Block until peer cumulatively acks `next_send_seq_-1`.
    const uint32_t need = next_send_seq_ - 1;
    return progress_until([&]{ return last_acked_seq_ >= need || !alive_; }, timeout_ms);
}

RdmaResult RdmaConnection::send_data(const void * data, size_t len, int timeout_ms) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!alive_) return RdmaResult::Closed;

    const uint32_t msg_id = next_msg_id_++;
    const uint32_t per    = payload_per_frame();
    const uint32_t n_frames = per ? (uint32_t) ((len + per - 1) / per) : 1;

    size_t off = 0;
    for (uint32_t i = 0; i < n_frames; i++) {
        RdmaResult r = progress_until([&]{
            for (auto & s : send_slots_) {
                if (!s.in_flight && s.payload_len == 0) return peer_credits_ > 0;
            }
            return false;
        }, timeout_ms);
        if (r != RdmaResult::Ok) {
            if (r == RdmaResult::Timeout) stats_.credit_stalls++;
            return r;
        }
        uint32_t idx = 0;
        for (; idx < send_window_; idx++) if (!send_slots_[idx].in_flight && send_slots_[idx].payload_len == 0) break;

        const uint16_t this_len = (uint16_t) std::min<size_t>(per, len - off);
        uint8_t * slot = send_buf_ + (size_t) idx * frame_size_;
        RdmaFrame h = {};
        h.msg_id = msg_id;
        h.seq    = next_send_seq_++;
        h.ack    = expected_recv_seq_ - 1;
        h.kind   = RDMA_KIND_DATA;
        h.flags  = (i == 0 ? RDMA_FLAG_FIRST : 0) | (i == n_frames - 1 ? RDMA_FLAG_LAST : 0);
        h.len    = this_len;
        h.payload_offset_or_total = (i == 0) ? (uint64_t) len : (uint64_t) off;
        std::memcpy(slot, &h, sizeof(h));
        if (this_len) std::memcpy(slot + sizeof(RdmaFrame), (const uint8_t *) data + off, this_len);

        send_slots_[idx].in_flight   = true;
        send_slots_[idx].acked       = false;
        send_slots_[idx].payload_len = sizeof(RdmaFrame) + this_len;
        send_slots_[idx].seq         = h.seq;
        send_slots_[idx].retransmits = 0;
        send_slots_[idx].last_post_ms = now_ms();
        if (!post_send_slot(idx, (uint16_t) send_slots_[idx].payload_len)) {
            send_slots_[idx] = SendSlot{};
            return RdmaResult::Dead;
        }
        in_flight_count_++;
        if (peer_credits_) peer_credits_--;
        off += this_len;
    }
    const uint32_t need = next_send_seq_ - 1;
    return progress_until([&]{ return last_acked_seq_ >= need || !alive_; }, timeout_ms);
}

RdmaResult RdmaConnection::recv_ctrl(std::vector<uint8_t> * out, int timeout_ms) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!alive_) return RdmaResult::Closed;
    RdmaResult r = progress_until([&]{ return !ctrl_ready_.empty() || !alive_; }, timeout_ms);
    if (r != RdmaResult::Ok) return r;
    if (ctrl_ready_.empty()) return RdmaResult::Dead;
    *out = std::move(ctrl_ready_.front());
    ctrl_ready_.erase(ctrl_ready_.begin());
    return RdmaResult::Ok;
}

RdmaResult RdmaConnection::recv_data(void * dest, size_t expected_len, int timeout_ms) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!alive_) return RdmaResult::Closed;
    if (data_recv_.active) return RdmaResult::ProtocolError;
    data_recv_.active = false;          // will become true on FIRST frame
    data_recv_.dest   = dest;
    data_recv_.total  = expected_len;
    data_recv_.copied = 0;
    data_recv_.msg_id = 0;
    data_recv_pending_complete_ = false;
    RdmaResult r = progress_until([&]{
        return data_recv_pending_complete_ || !alive_;
    }, timeout_ms);
    if (r != RdmaResult::Ok) {
        // Leave dest/total cleared so next call can retry; abort streaming.
        data_recv_ = DataRecvState{};
        return r;
    }
    if (data_recv_.copied != expected_len) {
        data_recv_ = DataRecvState{};
        return RdmaResult::ProtocolError;
    }
    data_recv_ = DataRecvState{};
    data_recv_pending_complete_ = false;
    return RdmaResult::Ok;
}

} // namespace ggml_tb_rdma

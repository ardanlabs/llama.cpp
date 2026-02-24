#pragma once

// Apple Thunderbolt UC SEND/RECV transport on top of `librdma.dylib`.
//
// One `RdmaConnection` owns: one QP, one CQ, one send MR (W * frame_size
// bytes), one recv MR (R * frame_size bytes) and the slot pools indexing
// into them. It speaks the RdmaFrame format defined in `protocol.h` and
// implements go-back-N reliability over UC.
//
// The public API exposes:
//   - bring_up()         — opens device, creates PD/CQ/QP, builds MRs.
//   - exchange_qp_info() — caller-driven over the still-open TCP socket.
//   - finalize()         — modify_qp to RTR/RTS, post all recvs.
//   - readiness_ping()   — first round-trip over RDMA, used as the
//                          go/no-go signal for the TCP-fallback decision.
//   - send_message()/recv_message_ctrl()/recv_message_data() — request
//                          framing on the cooked channel.
//
// All blocking methods take a timeout. Loss/dup tests force progress via
// `kick_retransmit()` / `set_drop_filter()` (only enabled with
// `GGML_TB_RDMA_FAULT_INJECT`).

#include "ibv.h"
#include "protocol.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ggml_tb_rdma {

struct RdmaStats {
    uint64_t frames_sent       = 0;
    uint64_t frames_received   = 0;
    uint64_t bytes_sent        = 0;
    uint64_t bytes_received    = 0;
    uint64_t retransmits       = 0;
    uint64_t credit_stalls     = 0;
    uint64_t qp_cq_errors      = 0;
    uint64_t flow_only_frames  = 0;
};

enum class RdmaResult {
    Ok,
    Timeout,
    Dead,            // connection declared dead (retransmit exhausted)
    ProtocolError,   // peer sent a malformed frame
    Closed,
};

class RdmaConnection {
public:
    RdmaConnection() = default;
    ~RdmaConnection();

    RdmaConnection(const RdmaConnection &)             = delete;
    RdmaConnection & operator=(const RdmaConnection &) = delete;

    // ---- Bring-up steps (driven from Transport) ------------------------

    // Step 1: open device, alloc PD, create CQ + QP, register MRs, transition
    // QP to INIT, post all recv WRs. `tcp_fd` is used only to derive the
    // local IP for GID matching (the caller still owns the fd).
    bool bring_up(int tcp_fd, std::string * err);

    // Local QP info to send to peer over TCP.
    MsgQPInfo local_qp_info() const;

    // Step 2: caller hands us peer's QP info; we modify_qp INIT → RTR → RTS.
    bool finalize(const MsgQPInfo & peer, std::string * err);

    // Step 3: tiny SEND/RECV round-trip over RDMA. Returns true if the
    // exchanged nonce round-trips byte-exact within `timeout_ms`. Caller
    // typically performs this AFTER both sides have finalized.
    bool readiness_ping(bool is_server, std::string * err, int timeout_ms = 5000);

    // Tear everything down cleanly: drain CQ, modify_qp(ERR), destroy.
    void close();

    bool alive() const { return alive_; }
    uint32_t frame_size() const { return frame_size_; }
    uint32_t payload_per_frame() const {
        return frame_size_ > sizeof(RdmaFrame) ? frame_size_ - (uint32_t) sizeof(RdmaFrame) : 0;
    }
    const RdmaStats & stats() const { return stats_; }

    // ---- Reliable message API (M2b/M2c) --------------------------------

    // Send a CONTROL message (cap MAX_CONTROL_FRAME_BYTES).
    RdmaResult send_ctrl(const void * data, size_t len, int timeout_ms);

    // Send a multi-part CONTROL message (saves copies).
    struct Slice { const void * data; size_t size; };
    RdmaResult send_ctrl_parts(const std::vector<Slice> & parts, int timeout_ms);

    // Send a DATA message of length `len` (no cap from the framework; the
    // protocol layer is responsible for sane upper bounds).
    RdmaResult send_data(const void * data, size_t len, int timeout_ms);

    // Receive the next CONTROL message into `out` (resized to message size).
    RdmaResult recv_ctrl(std::vector<uint8_t> * out, int timeout_ms);

    // Receive the next DATA message, streaming directly into `dest`. The
    // protocol layer pre-declares the expected total via `expected_len`;
    // the engine asserts `total == expected_len` from the FIRST frame and
    // never allocates a full reassembly buffer.
    RdmaResult recv_data(void * dest, size_t expected_len, int timeout_ms);

    // ---- Fault injection (test-only) -----------------------------------

#ifdef GGML_TB_RDMA_FAULT_INJECT
    // Drop every Nth outbound frame (N=0 disables).
    void set_drop_every(uint32_t n) { drop_every_ = n; }
#endif

private:
    // --- helpers (see rdma.cpp) ---
    bool open_device(std::string * err);
    bool register_mrs(std::string * err);
    bool create_qp(std::string * err);
    bool transition_to_init(std::string * err);
    bool transition_to_rtr(const MsgQPInfo & peer, std::string * err);
    bool transition_to_rts(std::string * err);
    bool post_initial_recvs(std::string * err);
    void drain_and_destroy();

    // Send/recv WR posting (raw — caller-owned wr_ids).
    bool post_recv_slot(uint32_t idx);
    bool post_send_slot(uint32_t idx, uint16_t payload_len);

    // Progress engine.
    using Predicate = std::function<bool()>;
    RdmaResult progress_until(const Predicate & p, int timeout_ms);
    void poll_completions();
    void maybe_emit_ack();
    void maybe_retransmit();

    // Reassembly bookkeeping.
    void on_recv_frame(const RdmaFrame & h, const uint8_t * payload, uint32_t idx);

    // ---- Members ----
    bool alive_ = false;

    // ibv handles
    struct ibv_context * ctx_   = nullptr;
    struct ibv_pd      * pd_    = nullptr;
    struct ibv_cq      * cq_    = nullptr;
    struct ibv_qp      * qp_    = nullptr;
    uint8_t              port_num_  = 1;
    uint8_t              gid_index_ = 0;
    union ibv_gid        local_gid_ = {};
    uint16_t             local_lid_ = 0;
    uint32_t             local_psn_ = 0;
    uint8_t              path_mtu_  = 0;   // enum ibv_mtu

    // Slot pools — page-aligned, posix_memalign-backed (TN3205 requirement;
    // Apple's Thunderbolt IOMMU only maps page-aligned regions).
    uint32_t        frame_size_  = 0;
    uint32_t        send_window_ = 0;
    uint32_t        recv_credits_ = 0;
    uint8_t       * send_buf_   = nullptr;
    size_t          send_buf_sz_ = 0;
    uint8_t       * recv_buf_   = nullptr;
    size_t          recv_buf_sz_ = 0;
    struct ibv_mr * send_mr_    = nullptr;
    struct ibv_mr * recv_mr_    = nullptr;

    // Send slot bookkeeping (W entries).
    struct SendSlot {
        bool      in_flight   = false;
        bool      acked       = false;
        uint16_t  payload_len = 0;          // includes RdmaFrame header
        uint32_t  seq         = 0;
        int       retransmits = 0;
        int64_t   last_post_ms = 0;
    };
    std::vector<SendSlot> send_slots_;
    uint32_t              next_send_slot_ = 0;

    // Reliability state.
    uint32_t next_send_seq_ = 1;     // monotonic per direction
    uint32_t expected_recv_seq_ = 1; // next seq we expect from peer
    uint32_t last_acked_seq_   = 0;  // largest peer seq we've cumulatively acked
    uint32_t peer_credits_     = 0;  // current peer-side credit budget
    uint32_t peer_credits_max_ = 0;  // R from peer hello (cap for replenishment)
    uint32_t in_flight_count_  = 0;
    int64_t  last_ack_emit_ms_ = 0;
    bool     ack_pending_      = false; // expected_seq advanced since last emit

    // CTRL reassembly: per-msg_id buffer until LAST arrives (cap 1 MiB).
    struct CtrlReassembly {
        uint64_t             total = 0;
        std::vector<uint8_t> bytes;
        bool                 complete = false;
    };
    std::unordered_map<uint32_t, CtrlReassembly> ctrl_pending_;
    std::vector<std::vector<uint8_t>>            ctrl_ready_;  // FIFO of completed CTRL messages

    // DATA streaming state: at most one DATA recv in progress at a time
    // (the protocol layer is request/response so this matches reality).
    struct DataRecvState {
        bool   active   = false;
        void * dest     = nullptr;
        uint64_t total  = 0;
        uint64_t copied = 0;
        uint32_t msg_id = 0;
    } data_recv_;

    bool data_recv_pending_complete_ = false;

    // Outbound msg-id counter.
    uint32_t next_msg_id_ = 1;

    // Local IP discovered from tcp_fd (for GID matching).
    std::string local_ip_;

    RdmaStats stats_{};

    // Single mutex around the public API (single-threaded progress engine).
    std::mutex mu_;

#ifdef GGML_TB_RDMA_FAULT_INJECT
    uint32_t drop_every_ = 0;
    uint64_t out_frame_counter_ = 0;
#endif
};

// ---- env-var helpers used by both rdma.cpp and transport.cpp ----------

bool env_truthy(const char * name);
int  env_int(const char * name, int defv);

} // namespace ggml_tb_rdma

#include "transport.h"

#include "ibv.h"
#include "rdma.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ggml_tb_rdma {

Transport::Transport()  = default;
Transport::~Transport() { close(); }

// -----------------------------------------------------------------------------
// Low-level frame I/O
// -----------------------------------------------------------------------------

RequestStatus Transport::send_frame_raw(TbRdmaCmd cmd, const void * payload, size_t size) {
    if (fd_ < 0) return RequestStatus::Closed;
    if (size > 0xFFFFFFFFu) {
        close_unlocked();
        return RequestStatus::ProtocolError;
    }

    MsgHeader hdr = {};
    hdr.cmd          = static_cast<uint8_t>(cmd);
    hdr.payload_size = static_cast<uint32_t>(size);

    IoResult r = tcp_send_all(fd_, &hdr, sizeof(hdr));
    if (r != IoResult::Ok) {
        close_unlocked();
        return RequestStatus::IoError;
    }
    if (size > 0) {
        r = tcp_send_all(fd_, payload, size);
        if (r != IoResult::Ok) {
            close_unlocked();
            return RequestStatus::IoError;
        }
    }
    stats_.frames_sent += 1;
    stats_.bytes_sent  += sizeof(hdr) + size;
    return RequestStatus::Ok;
}

RequestStatus Transport::send_frame_parts(TbRdmaCmd cmd, const std::vector<Slice> & parts) {
    if (fd_ < 0) return RequestStatus::Closed;

    size_t total = 0;
    for (const auto & p : parts) total += p.size;
    if (total > 0xFFFFFFFFu) {
        close_unlocked();
        return RequestStatus::ProtocolError;
    }

    MsgHeader hdr = {};
    hdr.cmd          = static_cast<uint8_t>(cmd);
    hdr.payload_size = static_cast<uint32_t>(total);

    IoResult r = tcp_send_all(fd_, &hdr, sizeof(hdr));
    if (r != IoResult::Ok) {
        close_unlocked();
        return RequestStatus::IoError;
    }
    for (const auto & p : parts) {
        if (p.size == 0) continue;
        r = tcp_send_all(fd_, p.data, p.size);
        if (r != IoResult::Ok) {
            close_unlocked();
            return RequestStatus::IoError;
        }
    }
    stats_.frames_sent += 1;
    stats_.bytes_sent  += sizeof(hdr) + total;
    return RequestStatus::Ok;
}

RequestStatus Transport::recv_frame_header(MsgHeader * hdr_out) {
    if (fd_ < 0) return RequestStatus::Closed;

    MsgHeader hdr = {};
    IoResult r = tcp_recv_all(fd_, &hdr, sizeof(hdr));
    if (r == IoResult::Closed) {
        close_unlocked();
        return RequestStatus::Disconnected;
    }
    if (r != IoResult::Ok) {
        close_unlocked();
        return RequestStatus::IoError;
    }
    if (hdr.flags != 0 || hdr.reserved != 0) {
        close_unlocked();
        return RequestStatus::ProtocolError;
    }
    *hdr_out = hdr;
    stats_.frames_received += 1;
    stats_.bytes_received  += sizeof(hdr);
    return RequestStatus::Ok;
}

RequestStatus Transport::recv_frame_payload(void * buf, size_t size) {
    if (fd_ < 0) return RequestStatus::Closed;
    if (size == 0) return RequestStatus::Ok;

    IoResult r = tcp_recv_all(fd_, buf, size);
    if (r == IoResult::Closed) {
        // Mid-frame close is an error, not an orderly disconnect.
        close_unlocked();
        return RequestStatus::IoError;
    }
    if (r != IoResult::Ok) {
        close_unlocked();
        return RequestStatus::IoError;
    }
    stats_.bytes_received += size;
    return RequestStatus::Ok;
}

// -----------------------------------------------------------------------------
// Hello
// -----------------------------------------------------------------------------

static bool features_ok(uint32_t peer_features, uint32_t peer_required, uint32_t local_features, std::string * err) {
    const uint32_t needed_by_peer = peer_required & ~local_features;
    if (needed_by_peer != 0) {
        if (err) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                          "peer requires features 0x%08x not advertised locally", needed_by_peer);
            *err = buf;
        }
        return false;
    }
    const uint32_t needed_by_local = GGML_TB_RDMA_FEATURE_FRAMED_V2 & ~peer_features;
    if (needed_by_local != 0) {
        if (err) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                          "peer missing required features 0x%08x", needed_by_local);
            *err = buf;
        }
        return false;
    }
    return true;
}

bool Transport::do_hello_exchange_client(std::string * err) {
    MsgHello tx = {};
    tx.version_major = GGML_TB_RDMA_PROTO_VERSION_MAJOR;
    tx.version_minor = 0;
    tx.features      = GGML_TB_RDMA_FEATURE_FRAMED_V2;  // SEND_RECV added by M2a
    tx.required      = GGML_TB_RDMA_FEATURE_FRAMED_V2;

    RequestStatus s = send_frame_raw(CMD_HELLO, &tx, sizeof(tx));
    if (s != RequestStatus::Ok) {
        if (err) *err = "hello: failed to send";
        return false;
    }

    MsgHeader hdr = {};
    s = recv_frame_header(&hdr);
    if (s != RequestStatus::Ok) {
        if (err) *err = "hello: failed to read response header";
        return false;
    }
    if (hdr.cmd != CMD_HELLO || hdr.payload_size != sizeof(MsgHello)) {
        close_unlocked();
        if (err) *err = "hello: bad response framing";
        return false;
    }

    MsgHello rx = {};
    s = recv_frame_payload(&rx, sizeof(rx));
    if (s != RequestStatus::Ok) {
        if (err) *err = "hello: failed to read response payload";
        return false;
    }

    if (rx.version_major != GGML_TB_RDMA_PROTO_VERSION_MAJOR) {
        close_unlocked();
        if (err) {
            char buf[96];
            std::snprintf(buf, sizeof(buf),
                          "hello: protocol major %u != %u",
                          rx.version_major, GGML_TB_RDMA_PROTO_VERSION_MAJOR);
            *err = buf;
        }
        return false;
    }
    if (!features_ok(rx.features, rx.required, tx.features, err)) {
        close_unlocked();
        return false;
    }
    return true;
}

bool Transport::do_hello_exchange_server(std::string * err) {
    MsgHeader hdr = {};
    RequestStatus s = recv_frame_header(&hdr);
    if (s != RequestStatus::Ok) {
        if (err) *err = "hello: failed to read request header";
        return false;
    }
    if (hdr.cmd != CMD_HELLO || hdr.payload_size != sizeof(MsgHello)) {
        close_unlocked();
        if (err) *err = "hello: bad request framing";
        return false;
    }

    MsgHello rx = {};
    s = recv_frame_payload(&rx, sizeof(rx));
    if (s != RequestStatus::Ok) {
        if (err) *err = "hello: failed to read request payload";
        return false;
    }

    if (rx.version_major != GGML_TB_RDMA_PROTO_VERSION_MAJOR) {
        close_unlocked();
        if (err) {
            char buf[96];
            std::snprintf(buf, sizeof(buf),
                          "hello: protocol major %u != %u",
                          rx.version_major, GGML_TB_RDMA_PROTO_VERSION_MAJOR);
            *err = buf;
        }
        return false;
    }

    MsgHello tx = {};
    tx.version_major = GGML_TB_RDMA_PROTO_VERSION_MAJOR;
    tx.version_minor = 0;
    tx.features      = GGML_TB_RDMA_FEATURE_FRAMED_V2;
    tx.required      = GGML_TB_RDMA_FEATURE_FRAMED_V2;

    if (!features_ok(rx.features, rx.required, tx.features, err)) {
        close_unlocked();
        return false;
    }

    s = send_frame_raw(CMD_HELLO, &tx, sizeof(tx));
    if (s != RequestStatus::Ok) {
        if (err) *err = "hello: failed to send response";
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Bring-up
// -----------------------------------------------------------------------------

bool Transport::client_connect(const Endpoint & ep,
                               const std::string & /*rdma_device*/,
                               std::string * err) {
    std::lock_guard<std::mutex> lock(mu_);
    if (fd_ >= 0) {
        if (err) *err = "transport already connected";
        return false;
    }
    int fd = tcp_connect(ep, err);
    if (fd < 0) return false;
    fd_        = fd;
    mode_      = TransportMode::Tcp;
    is_server_ = false;
    if (!do_hello_exchange_client(err)) {
        return false;
    }
    std::string promote_err;
    if (!try_promote_to_rdma_client(&promote_err)) {
        // try_promote_* logs internally; if it returned false and set
        // promote_err, that's the hard-error case.
        if (!promote_err.empty()) {
            if (err) *err = promote_err;
            close_unlocked();
            return false;
        }
    }
    return true;
}

bool Transport::server_accept(int accepted_fd,
                              const std::string & /*rdma_device*/,
                              std::string * err) {
    std::lock_guard<std::mutex> lock(mu_);
    if (fd_ >= 0) {
        if (err) *err = "transport already connected";
        return false;
    }
    fd_        = accepted_fd;
    mode_      = TransportMode::Tcp;
    is_server_ = true;
    if (!do_hello_exchange_server(err)) {
        return false;
    }
    std::string promote_err;
    if (!try_promote_to_rdma_server(&promote_err)) {
        if (!promote_err.empty()) {
            if (err) *err = promote_err;
            close_unlocked();
            return false;
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
// Teardown
// -----------------------------------------------------------------------------

void Transport::close() {
    std::lock_guard<std::mutex> lock(mu_);
    if (fd_ < 0 && !rdma_) return;
    // Best-effort goodbye; ignore failure (peer may already be gone).
    if (!closed_after_error_) {
        if (mode_ == TransportMode::Rdma && rdma_ && rdma_->alive()) {
            uint8_t goodbye_hdr[sizeof(MsgHeader)] = {};
            MsgHeader h = {}; h.cmd = CMD_GOODBYE; h.payload_size = 0;
            std::memcpy(goodbye_hdr, &h, sizeof(h));
            (void) rdma_->send_ctrl(goodbye_hdr, sizeof(goodbye_hdr),
                                    TB_RDMA_DEFAULT_OP_TIMEOUT_MS);
        } else if (fd_ >= 0) {
            (void)send_frame_raw(CMD_GOODBYE, nullptr, 0);
        }
    }
    close_unlocked();
}

void Transport::close_unlocked() {
    if (rdma_) {
        update_stats_from_rdma();
        rdma_->close();
        rdma_.reset();
    }
    if (fd_ >= 0) {
        tcp_close(fd_);
        fd_ = -1;
    }
    closed_after_error_ = true;
}

// -----------------------------------------------------------------------------
// request_raw / request_var
// -----------------------------------------------------------------------------

RequestStatus Transport::request_raw(TbRdmaCmd cmd,
                                     const void * req, size_t req_size,
                                     TbRdmaCmd expected_rsp_cmd,
                                     void * rsp, size_t rsp_size,
                                     uint16_t * peer_err_code,
                                     std::string * peer_err_detail) {
    std::lock_guard<std::mutex> lock(mu_);
    if (mode_ == TransportMode::Rdma) {
        return rdma_request_raw(cmd, req, req_size, expected_rsp_cmd, rsp, rsp_size,
                                peer_err_code, peer_err_detail);
    }
    if (fd_ < 0) return RequestStatus::Closed;

    RequestStatus s = send_frame_raw(cmd, req, req_size);
    if (s != RequestStatus::Ok) return s;

    MsgHeader hdr = {};
    s = recv_frame_header(&hdr);
    if (s != RequestStatus::Ok) return s;

    // Allow peer to surface an error in place of the expected response.
    if (hdr.cmd == CMD_ERROR) {
        if (hdr.payload_size < sizeof(MsgError) || hdr.payload_size > MAX_CONTROL_FRAME_BYTES) {
            close_unlocked();
            return RequestStatus::ProtocolError;
        }
        MsgError err_payload = {};
        s = recv_frame_payload(&err_payload, sizeof(err_payload));
        if (s != RequestStatus::Ok) return s;

        const uint32_t detail = hdr.payload_size - static_cast<uint32_t>(sizeof(MsgError));
        if (detail != err_payload.detail_len) {
            close_unlocked();
            return RequestStatus::ProtocolError;
        }
        std::vector<uint8_t> detail_bytes(detail);
        if (detail > 0) {
            s = recv_frame_payload(detail_bytes.data(), detail);
            if (s != RequestStatus::Ok) return s;
        }
        if (peer_err_code)   *peer_err_code   = err_payload.code;
        if (peer_err_detail) peer_err_detail->assign(detail_bytes.begin(), detail_bytes.end());
        close_unlocked();
        return RequestStatus::PeerError;
    }

    if (hdr.cmd != static_cast<uint8_t>(expected_rsp_cmd) || hdr.payload_size != rsp_size) {
        close_unlocked();
        return RequestStatus::ProtocolError;
    }
    s = recv_frame_payload(rsp, rsp_size);
    if (s != RequestStatus::Ok) return s;
    return RequestStatus::Ok;
}

RequestStatus Transport::request_var(TbRdmaCmd cmd,
                                     const std::vector<Slice> & req_parts,
                                     TbRdmaCmd expected_rsp_cmd,
                                     std::vector<uint8_t> * rsp_out,
                                     uint32_t max_rsp_bytes,
                                     uint16_t * peer_err_code,
                                     std::string * peer_err_detail) {
    std::lock_guard<std::mutex> lock(mu_);
    if (mode_ == TransportMode::Rdma) {
        return rdma_request_var(cmd, req_parts, expected_rsp_cmd, rsp_out, max_rsp_bytes,
                                peer_err_code, peer_err_detail);
    }
    if (fd_ < 0) return RequestStatus::Closed;

    RequestStatus s = send_frame_parts(cmd, req_parts);
    if (s != RequestStatus::Ok) return s;

    MsgHeader hdr = {};
    s = recv_frame_header(&hdr);
    if (s != RequestStatus::Ok) return s;

    if (hdr.cmd == CMD_ERROR) {
        if (hdr.payload_size < sizeof(MsgError) || hdr.payload_size > MAX_CONTROL_FRAME_BYTES) {
            close_unlocked();
            return RequestStatus::ProtocolError;
        }
        MsgError err_payload = {};
        s = recv_frame_payload(&err_payload, sizeof(err_payload));
        if (s != RequestStatus::Ok) return s;
        const uint32_t detail = hdr.payload_size - static_cast<uint32_t>(sizeof(MsgError));
        if (detail != err_payload.detail_len) {
            close_unlocked();
            return RequestStatus::ProtocolError;
        }
        std::vector<uint8_t> detail_bytes(detail);
        if (detail > 0) {
            s = recv_frame_payload(detail_bytes.data(), detail);
            if (s != RequestStatus::Ok) return s;
        }
        if (peer_err_code)   *peer_err_code   = err_payload.code;
        if (peer_err_detail) peer_err_detail->assign(detail_bytes.begin(), detail_bytes.end());
        close_unlocked();
        return RequestStatus::PeerError;
    }

    if (hdr.cmd != static_cast<uint8_t>(expected_rsp_cmd) || hdr.payload_size > max_rsp_bytes) {
        close_unlocked();
        return RequestStatus::ProtocolError;
    }
    rsp_out->resize(hdr.payload_size);
    if (hdr.payload_size > 0) {
        s = recv_frame_payload(rsp_out->data(), hdr.payload_size);
        if (s != RequestStatus::Ok) return s;
    }
    return RequestStatus::Ok;
}

// -----------------------------------------------------------------------------
// Server primitives
// -----------------------------------------------------------------------------

RequestStatus Transport::recv_request(MsgHeader * hdr_out,
                                      std::vector<uint8_t> * payload_out,
                                      uint32_t max_payload_bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    if (mode_ == TransportMode::Rdma) {
        return rdma_recv_request(hdr_out, payload_out, max_payload_bytes);
    }
    if (fd_ < 0) return RequestStatus::Closed;

    MsgHeader hdr = {};
    RequestStatus s = recv_frame_header(&hdr);
    if (s != RequestStatus::Ok) return s;

    if (hdr.payload_size > max_payload_bytes) {
        close_unlocked();
        return RequestStatus::ProtocolError;
    }
    payload_out->resize(hdr.payload_size);
    if (hdr.payload_size > 0) {
        s = recv_frame_payload(payload_out->data(), hdr.payload_size);
        if (s != RequestStatus::Ok) return s;
    }
    *hdr_out = hdr;
    return RequestStatus::Ok;
}

RequestStatus Transport::send_response_raw(TbRdmaCmd cmd, const void * payload, size_t size) {
    std::lock_guard<std::mutex> lock(mu_);
    if (mode_ == TransportMode::Rdma) return rdma_send_response_raw(cmd, payload, size);
    return send_frame_raw(cmd, payload, size);
}

RequestStatus Transport::send_response_var(TbRdmaCmd cmd, const std::vector<Slice> & parts) {
    std::lock_guard<std::mutex> lock(mu_);
    if (mode_ == TransportMode::Rdma) return rdma_send_response_var(cmd, parts);
    return send_frame_parts(cmd, parts);
}

void Transport::send_error_and_close(TbRdmaError code, const std::string & detail) {
    std::lock_guard<std::mutex> lock(mu_);
    if (fd_ < 0 && !rdma_) return;

    MsgError hdr = {};
    hdr.code       = static_cast<uint16_t>(code);
    hdr.detail_len = static_cast<uint32_t>(std::min<size_t>(detail.size(), MAX_CONTROL_FRAME_BYTES - sizeof(MsgError)));

    std::vector<Slice> parts = {
        { &hdr, sizeof(hdr) },
        { detail.data(), hdr.detail_len },
    };
    if (mode_ == TransportMode::Rdma) {
        (void) rdma_send_response_var(CMD_ERROR, parts);
    } else {
        (void) send_frame_parts(CMD_ERROR, parts);
    }
    close_unlocked();
}

// -----------------------------------------------------------------------------
// RDMA bring-up (M2a)
// -----------------------------------------------------------------------------

bool Transport::try_promote_to_rdma_client(std::string * err) {
    const bool force_tcp = env_truthy("GGML_TB_RDMA_FORCE_TCP")
                        || env_truthy("GGML_TB_RDMA_DISABLE");
    const bool require   = env_truthy("GGML_TB_RDMA_REQUIRE_RDMA");

    auto fallback = [&](const char * reason) -> bool {
        std::fprintf(stderr, "[tb-rdma] RDMA setup failed: %s; falling back to TCP\n", reason);
        stats_.fallback_events++;
        if (require) { if (err) *err = std::string("RDMA required but ") + reason; return false; }
        return true;  // stay in TCP mode (returns false from caller's perspective)
    };

    if (force_tcp) {
        std::fprintf(stderr, "[tb-rdma] RDMA disabled by env; using TCP\n");
        return false;
    }
    if (!ibv().available || !has_rdma_devices()) {
        if (!fallback("librdma.dylib not available or no RDMA devices")) return false;
        return false;
    }

    rdma_.reset(new RdmaConnection());
    std::string rerr;
    if (!rdma_->bring_up(fd_, &rerr)) {
        rdma_.reset();
        if (!fallback(rerr.c_str())) return false;
        return false;
    }

    // Exchange QP info over the still-open TCP socket.
    MsgQPInfo my_info = rdma_->local_qp_info();
    if (send_frame_raw(CMD_QP_INFO, &my_info, sizeof(my_info)) != RequestStatus::Ok) {
        rdma_.reset();
        if (!fallback("failed to send QP info")) return false;
        return false;
    }
    MsgHeader hdr = {};
    if (recv_frame_header(&hdr) != RequestStatus::Ok ||
        hdr.cmd != CMD_QP_INFO ||
        hdr.payload_size != sizeof(MsgQPInfo)) {
        rdma_.reset();
        if (!fallback("failed to recv QP info")) return false;
        return false;
    }
    MsgQPInfo peer = {};
    if (recv_frame_payload(&peer, sizeof(peer)) != RequestStatus::Ok) {
        rdma_.reset();
        if (!fallback("failed to recv QP info payload")) return false;
        return false;
    }

    if (!rdma_->finalize(peer, &rerr)) {
        rdma_.reset();
        if (!fallback(rerr.c_str())) return false;
        return false;
    }

    if (!rdma_->readiness_ping(/*is_server=*/false, &rerr,
                               env_int("GGML_TB_RDMA_PING_TIMEOUT_MS", 5000))) {
        rdma_.reset();
        if (!fallback(rerr.c_str())) return false;
        return false;
    }

    // Success: tear down TCP and switch to RDMA mode.
    tcp_close(fd_);
    fd_   = -1;
    mode_ = TransportMode::Rdma;
    std::fprintf(stderr, "[tb-rdma] RDMA bring-up succeeded; closed TCP bootstrap\n");
    return true;
}

bool Transport::try_promote_to_rdma_server(std::string * err) {
    const bool force_tcp = env_truthy("GGML_TB_RDMA_FORCE_TCP")
                        || env_truthy("GGML_TB_RDMA_DISABLE");
    const bool require   = env_truthy("GGML_TB_RDMA_REQUIRE_RDMA");

    auto fallback = [&](const char * reason) -> bool {
        std::fprintf(stderr, "[tb-rdma] RDMA setup failed: %s; falling back to TCP\n", reason);
        stats_.fallback_events++;
        if (require) { if (err) *err = std::string("RDMA required but ") + reason; return false; }
        return true;
    };

    if (force_tcp) {
        std::fprintf(stderr, "[tb-rdma] RDMA disabled by env; using TCP\n");
        return false;
    }
    if (!ibv().available || !has_rdma_devices()) {
        if (!fallback("librdma.dylib not available or no RDMA devices")) return false;
        return false;
    }

    rdma_.reset(new RdmaConnection());
    std::string rerr;
    if (!rdma_->bring_up(fd_, &rerr)) {
        rdma_.reset();
        if (!fallback(rerr.c_str())) return false;
        return false;
    }

    // Server reads client's QP info first, then sends its own.
    MsgHeader hdr = {};
    if (recv_frame_header(&hdr) != RequestStatus::Ok ||
        hdr.cmd != CMD_QP_INFO ||
        hdr.payload_size != sizeof(MsgQPInfo)) {
        rdma_.reset();
        if (!fallback("failed to recv QP info")) return false;
        return false;
    }
    MsgQPInfo peer = {};
    if (recv_frame_payload(&peer, sizeof(peer)) != RequestStatus::Ok) {
        rdma_.reset();
        if (!fallback("failed to recv QP info payload")) return false;
        return false;
    }
    MsgQPInfo my_info = rdma_->local_qp_info();
    if (send_frame_raw(CMD_QP_INFO, &my_info, sizeof(my_info)) != RequestStatus::Ok) {
        rdma_.reset();
        if (!fallback("failed to send QP info")) return false;
        return false;
    }
    if (!rdma_->finalize(peer, &rerr)) {
        rdma_.reset();
        if (!fallback(rerr.c_str())) return false;
        return false;
    }
    if (!rdma_->readiness_ping(/*is_server=*/true, &rerr,
                               env_int("GGML_TB_RDMA_PING_TIMEOUT_MS", 5000))) {
        rdma_.reset();
        if (!fallback(rerr.c_str())) return false;
        return false;
    }

    tcp_close(fd_);
    fd_   = -1;
    mode_ = TransportMode::Rdma;
    std::fprintf(stderr, "[tb-rdma] RDMA bring-up succeeded; closed TCP bootstrap\n");
    return true;
}

void Transport::update_stats_from_rdma() {
    if (!rdma_) return;
    const auto & rs = rdma_->stats();
    stats_.rdma_frames_sent     = rs.frames_sent;
    stats_.rdma_frames_received = rs.frames_received;
    stats_.rdma_retransmits     = rs.retransmits;
    stats_.rdma_credit_stalls   = rs.credit_stalls;
    stats_.rdma_qp_cq_errors    = rs.qp_cq_errors;
    stats_.rdma_flow_frames     = rs.flow_only_frames;
    // Mirror byte counters too.
    stats_.bytes_sent     = rs.bytes_sent;
    stats_.bytes_received = rs.bytes_received;
    stats_.frames_sent    = rs.frames_sent;
    stats_.frames_received= rs.frames_received;
}

// -----------------------------------------------------------------------------
// RDMA framing — map TCP-shaped frames onto CTRL messages.
// -----------------------------------------------------------------------------

RequestStatus Transport::rdma_request_raw(TbRdmaCmd cmd,
                                          const void * req, size_t req_size,
                                          TbRdmaCmd expected_rsp_cmd,
                                          void * rsp, size_t rsp_size,
                                          uint16_t * peer_err_code,
                                          std::string * peer_err_detail) {
    if (!rdma_ || !rdma_->alive()) return RequestStatus::Closed;

    // Send: [MsgHeader][req].
    MsgHeader h = {};
    h.cmd          = static_cast<uint8_t>(cmd);
    h.payload_size = static_cast<uint32_t>(req_size);

    std::vector<RdmaConnection::Slice> parts = {
        { &h, sizeof(h) },
        { req, req_size },
    };
    RdmaResult r = rdma_->send_ctrl_parts(parts, TB_RDMA_DEFAULT_OP_TIMEOUT_MS);
    update_stats_from_rdma();
    if (r != RdmaResult::Ok) { close_unlocked(); return RequestStatus::IoError; }

    std::vector<uint8_t> rx;
    r = rdma_->recv_ctrl(&rx, TB_RDMA_DEFAULT_OP_TIMEOUT_MS);
    update_stats_from_rdma();
    if (r != RdmaResult::Ok) { close_unlocked(); return RequestStatus::IoError; }
    if (rx.size() < sizeof(MsgHeader)) { close_unlocked(); return RequestStatus::ProtocolError; }

    MsgHeader rh = {}; std::memcpy(&rh, rx.data(), sizeof(rh));
    if (rh.cmd == CMD_ERROR) {
        if (rx.size() < sizeof(MsgHeader) + sizeof(MsgError)) { close_unlocked(); return RequestStatus::ProtocolError; }
        MsgError e = {}; std::memcpy(&e, rx.data() + sizeof(MsgHeader), sizeof(e));
        const uint32_t detail = rh.payload_size - static_cast<uint32_t>(sizeof(MsgError));
        if (rx.size() != sizeof(MsgHeader) + sizeof(MsgError) + detail) { close_unlocked(); return RequestStatus::ProtocolError; }
        if (peer_err_code)   *peer_err_code = e.code;
        if (peer_err_detail) peer_err_detail->assign(rx.begin() + sizeof(MsgHeader) + sizeof(MsgError), rx.end());
        close_unlocked();
        return RequestStatus::PeerError;
    }
    if (rh.cmd != static_cast<uint8_t>(expected_rsp_cmd) ||
        rh.payload_size != rsp_size ||
        rx.size() != sizeof(MsgHeader) + rsp_size) {
        close_unlocked();
        return RequestStatus::ProtocolError;
    }
    if (rsp_size) std::memcpy(rsp, rx.data() + sizeof(MsgHeader), rsp_size);
    return RequestStatus::Ok;
}

RequestStatus Transport::rdma_request_var(TbRdmaCmd cmd,
                                          const std::vector<Slice> & req_parts,
                                          TbRdmaCmd expected_rsp_cmd,
                                          std::vector<uint8_t> * rsp_out,
                                          uint32_t max_rsp_bytes,
                                          uint16_t * peer_err_code,
                                          std::string * peer_err_detail) {
    if (!rdma_ || !rdma_->alive()) return RequestStatus::Closed;

    size_t total = 0; for (auto & p : req_parts) total += p.size;
    if (total > MAX_CONTROL_FRAME_BYTES - sizeof(MsgHeader)) {
        close_unlocked();
        return RequestStatus::ProtocolError;
    }
    MsgHeader h = {};
    h.cmd          = static_cast<uint8_t>(cmd);
    h.payload_size = static_cast<uint32_t>(total);

    std::vector<RdmaConnection::Slice> sl;
    sl.reserve(req_parts.size() + 1);
    sl.push_back({ &h, sizeof(h) });
    for (auto & p : req_parts) sl.push_back({ p.data, p.size });

    RdmaResult r = rdma_->send_ctrl_parts(sl, TB_RDMA_DEFAULT_OP_TIMEOUT_MS);
    update_stats_from_rdma();
    if (r != RdmaResult::Ok) { close_unlocked(); return RequestStatus::IoError; }

    std::vector<uint8_t> rx;
    r = rdma_->recv_ctrl(&rx, TB_RDMA_DEFAULT_OP_TIMEOUT_MS);
    update_stats_from_rdma();
    if (r != RdmaResult::Ok) { close_unlocked(); return RequestStatus::IoError; }
    if (rx.size() < sizeof(MsgHeader)) { close_unlocked(); return RequestStatus::ProtocolError; }

    MsgHeader rh = {}; std::memcpy(&rh, rx.data(), sizeof(rh));
    if (rh.cmd == CMD_ERROR) {
        if (rx.size() < sizeof(MsgHeader) + sizeof(MsgError)) { close_unlocked(); return RequestStatus::ProtocolError; }
        MsgError e = {}; std::memcpy(&e, rx.data() + sizeof(MsgHeader), sizeof(e));
        const uint32_t detail = rh.payload_size - static_cast<uint32_t>(sizeof(MsgError));
        if (rx.size() != sizeof(MsgHeader) + sizeof(MsgError) + detail) { close_unlocked(); return RequestStatus::ProtocolError; }
        if (peer_err_code)   *peer_err_code = e.code;
        if (peer_err_detail) peer_err_detail->assign(rx.begin() + sizeof(MsgHeader) + sizeof(MsgError), rx.end());
        close_unlocked();
        return RequestStatus::PeerError;
    }
    if (rh.cmd != static_cast<uint8_t>(expected_rsp_cmd) ||
        rh.payload_size > max_rsp_bytes ||
        rx.size() != sizeof(MsgHeader) + rh.payload_size) {
        close_unlocked();
        return RequestStatus::ProtocolError;
    }
    rsp_out->assign(rx.begin() + sizeof(MsgHeader), rx.end());
    return RequestStatus::Ok;
}

RequestStatus Transport::rdma_recv_request(MsgHeader * hdr_out,
                                           std::vector<uint8_t> * payload_out,
                                           uint32_t max_payload_bytes) {
    if (!rdma_ || !rdma_->alive()) return RequestStatus::Closed;
    std::vector<uint8_t> rx;
    RdmaResult r = rdma_->recv_ctrl(&rx, /*infinite-ish*/ 24 * 60 * 60 * 1000);
    update_stats_from_rdma();
    if (r != RdmaResult::Ok) { close_unlocked(); return RequestStatus::IoError; }
    if (rx.size() < sizeof(MsgHeader)) { close_unlocked(); return RequestStatus::ProtocolError; }
    MsgHeader h = {}; std::memcpy(&h, rx.data(), sizeof(h));
    if (h.payload_size > max_payload_bytes ||
        rx.size() != sizeof(MsgHeader) + h.payload_size) {
        close_unlocked();
        return RequestStatus::ProtocolError;
    }
    *hdr_out = h;
    payload_out->assign(rx.begin() + sizeof(MsgHeader), rx.end());
    return RequestStatus::Ok;
}

RequestStatus Transport::rdma_send_response_raw(TbRdmaCmd cmd, const void * payload, size_t size) {
    if (!rdma_ || !rdma_->alive()) return RequestStatus::Closed;
    MsgHeader h = {};
    h.cmd          = static_cast<uint8_t>(cmd);
    h.payload_size = static_cast<uint32_t>(size);
    std::vector<RdmaConnection::Slice> sl = {
        { &h, sizeof(h) },
        { payload, size },
    };
    RdmaResult r = rdma_->send_ctrl_parts(sl, TB_RDMA_DEFAULT_OP_TIMEOUT_MS);
    update_stats_from_rdma();
    return r == RdmaResult::Ok ? RequestStatus::Ok : RequestStatus::IoError;
}

RequestStatus Transport::rdma_send_response_var(TbRdmaCmd cmd, const std::vector<Slice> & parts) {
    if (!rdma_ || !rdma_->alive()) return RequestStatus::Closed;
    size_t total = 0; for (auto & p : parts) total += p.size;
    MsgHeader h = {};
    h.cmd          = static_cast<uint8_t>(cmd);
    h.payload_size = static_cast<uint32_t>(total);
    std::vector<RdmaConnection::Slice> sl;
    sl.reserve(parts.size() + 1);
    sl.push_back({ &h, sizeof(h) });
    for (auto & p : parts) sl.push_back({ p.data, p.size });
    RdmaResult r = rdma_->send_ctrl_parts(sl, TB_RDMA_DEFAULT_OP_TIMEOUT_MS);
    update_stats_from_rdma();
    return r == RdmaResult::Ok ? RequestStatus::Ok : RequestStatus::IoError;
}

// -----------------------------------------------------------------------------
// Bulk request/response (M2c streaming)
// -----------------------------------------------------------------------------

RequestStatus Transport::request_bulk_send(TbRdmaCmd cmd,
                                           const void * req_ctrl, size_t req_ctrl_size,
                                           const void * body, size_t body_size,
                                           TbRdmaCmd expected_rsp_cmd,
                                           void * rsp, size_t rsp_size,
                                           uint16_t * peer_err_code,
                                           std::string * peer_err_detail) {
    std::lock_guard<std::mutex> lock(mu_);

    if (mode_ == TransportMode::Tcp) {
        if (fd_ < 0) return RequestStatus::Closed;
        const size_t total = req_ctrl_size + body_size;
        if (total > 0xFFFFFFFFu) { close_unlocked(); return RequestStatus::ProtocolError; }
        MsgHeader h = {};
        h.cmd          = static_cast<uint8_t>(cmd);
        h.payload_size = static_cast<uint32_t>(total);
        IoResult ir = tcp_send_all(fd_, &h, sizeof(h));
        if (ir != IoResult::Ok) { close_unlocked(); return RequestStatus::IoError; }
        if (req_ctrl_size) {
            ir = tcp_send_all(fd_, req_ctrl, req_ctrl_size);
            if (ir != IoResult::Ok) { close_unlocked(); return RequestStatus::IoError; }
        }
        if (body_size) {
            ir = tcp_send_all(fd_, body, body_size);
            if (ir != IoResult::Ok) { close_unlocked(); return RequestStatus::IoError; }
        }
        stats_.frames_sent += 1;
        stats_.bytes_sent  += sizeof(h) + total;
        // Receive fixed-size response.
        MsgHeader rh = {};
        RequestStatus s = recv_frame_header(&rh);
        if (s != RequestStatus::Ok) return s;
        if (rh.cmd == CMD_ERROR) {
            if (rh.payload_size < sizeof(MsgError) || rh.payload_size > MAX_CONTROL_FRAME_BYTES) {
                close_unlocked(); return RequestStatus::ProtocolError;
            }
            MsgError ep = {};
            s = recv_frame_payload(&ep, sizeof(ep));
            if (s != RequestStatus::Ok) return s;
            const uint32_t detail = rh.payload_size - static_cast<uint32_t>(sizeof(MsgError));
            std::vector<uint8_t> det(detail);
            if (detail > 0) {
                s = recv_frame_payload(det.data(), detail);
                if (s != RequestStatus::Ok) return s;
            }
            if (peer_err_code)   *peer_err_code   = ep.code;
            if (peer_err_detail) peer_err_detail->assign(det.begin(), det.end());
            close_unlocked();
            return RequestStatus::PeerError;
        }
        if (rh.cmd != static_cast<uint8_t>(expected_rsp_cmd) || rh.payload_size != rsp_size) {
            close_unlocked(); return RequestStatus::ProtocolError;
        }
        return recv_frame_payload(rsp, rsp_size);
    }

    // RDMA mode: CTRL[hdr + req_ctrl], then DATA[body], then recv CTRL[hdr + rsp].
    if (!rdma_ || !rdma_->alive()) return RequestStatus::Closed;
    MsgHeader h = {};
    h.cmd          = static_cast<uint8_t>(cmd);
    h.payload_size = static_cast<uint32_t>(req_ctrl_size + body_size);
    std::vector<RdmaConnection::Slice> ctrl_parts = {
        { &h, sizeof(h) },
        { req_ctrl, req_ctrl_size },
    };
    RdmaResult r = rdma_->send_ctrl_parts(ctrl_parts, TB_RDMA_DEFAULT_OP_TIMEOUT_MS);
    update_stats_from_rdma();
    if (r != RdmaResult::Ok) { close_unlocked(); return RequestStatus::IoError; }
    if (body_size) {
        r = rdma_->send_data(body, body_size, TB_RDMA_DEFAULT_OP_TIMEOUT_MS);
        update_stats_from_rdma();
        if (r != RdmaResult::Ok) { close_unlocked(); return RequestStatus::IoError; }
    }
    std::vector<uint8_t> rx;
    r = rdma_->recv_ctrl(&rx, TB_RDMA_DEFAULT_OP_TIMEOUT_MS);
    update_stats_from_rdma();
    if (r != RdmaResult::Ok) { close_unlocked(); return RequestStatus::IoError; }
    if (rx.size() < sizeof(MsgHeader)) { close_unlocked(); return RequestStatus::ProtocolError; }
    MsgHeader rh = {}; std::memcpy(&rh, rx.data(), sizeof(rh));
    if (rh.cmd == CMD_ERROR) {
        if (rx.size() < sizeof(MsgHeader) + sizeof(MsgError)) { close_unlocked(); return RequestStatus::ProtocolError; }
        MsgError ep = {}; std::memcpy(&ep, rx.data() + sizeof(MsgHeader), sizeof(ep));
        if (peer_err_code)   *peer_err_code = ep.code;
        if (peer_err_detail) peer_err_detail->assign(rx.begin() + sizeof(MsgHeader) + sizeof(MsgError), rx.end());
        close_unlocked();
        return RequestStatus::PeerError;
    }
    if (rh.cmd != static_cast<uint8_t>(expected_rsp_cmd) ||
        rh.payload_size != rsp_size ||
        rx.size() != sizeof(MsgHeader) + rsp_size) {
        close_unlocked();
        return RequestStatus::ProtocolError;
    }
    if (rsp_size) std::memcpy(rsp, rx.data() + sizeof(MsgHeader), rsp_size);
    return RequestStatus::Ok;
}

RequestStatus Transport::request_bulk_recv(TbRdmaCmd cmd,
                                           const void * req_ctrl, size_t req_ctrl_size,
                                           TbRdmaCmd expected_rsp_cmd,
                                           void * rsp, size_t rsp_size,
                                           void * body_dest, size_t body_size,
                                           uint16_t * peer_err_code,
                                           std::string * peer_err_detail) {
    std::lock_guard<std::mutex> lock(mu_);

    if (mode_ == TransportMode::Tcp) {
        if (fd_ < 0) return RequestStatus::Closed;
        MsgHeader qh = {};
        qh.cmd          = static_cast<uint8_t>(cmd);
        qh.payload_size = static_cast<uint32_t>(req_ctrl_size);
        IoResult ir = tcp_send_all(fd_, &qh, sizeof(qh));
        if (ir != IoResult::Ok) { close_unlocked(); return RequestStatus::IoError; }
        if (req_ctrl_size) {
            ir = tcp_send_all(fd_, req_ctrl, req_ctrl_size);
            if (ir != IoResult::Ok) { close_unlocked(); return RequestStatus::IoError; }
        }
        stats_.frames_sent += 1;
        stats_.bytes_sent  += sizeof(qh) + req_ctrl_size;

        MsgHeader rh = {};
        RequestStatus s = recv_frame_header(&rh);
        if (s != RequestStatus::Ok) return s;
        if (rh.cmd == CMD_ERROR) {
            if (rh.payload_size < sizeof(MsgError) || rh.payload_size > MAX_CONTROL_FRAME_BYTES) {
                close_unlocked(); return RequestStatus::ProtocolError;
            }
            MsgError ep = {};
            s = recv_frame_payload(&ep, sizeof(ep));
            if (s != RequestStatus::Ok) return s;
            const uint32_t detail = rh.payload_size - static_cast<uint32_t>(sizeof(MsgError));
            std::vector<uint8_t> det(detail);
            if (detail > 0) {
                s = recv_frame_payload(det.data(), detail);
                if (s != RequestStatus::Ok) return s;
            }
            if (peer_err_code)   *peer_err_code   = ep.code;
            if (peer_err_detail) peer_err_detail->assign(det.begin(), det.end());
            close_unlocked();
            return RequestStatus::PeerError;
        }
        if (rh.cmd != static_cast<uint8_t>(expected_rsp_cmd) ||
            rh.payload_size != rsp_size + body_size) {
            close_unlocked(); return RequestStatus::ProtocolError;
        }
        s = recv_frame_payload(rsp, rsp_size);
        if (s != RequestStatus::Ok) return s;
        if (body_size) {
            s = recv_frame_payload(body_dest, body_size);
            if (s != RequestStatus::Ok) return s;
        }
        return RequestStatus::Ok;
    }

    if (!rdma_ || !rdma_->alive()) return RequestStatus::Closed;
    MsgHeader qh = {};
    qh.cmd          = static_cast<uint8_t>(cmd);
    qh.payload_size = static_cast<uint32_t>(req_ctrl_size);
    std::vector<RdmaConnection::Slice> ctrl_parts = {
        { &qh, sizeof(qh) },
        { req_ctrl, req_ctrl_size },
    };
    RdmaResult r = rdma_->send_ctrl_parts(ctrl_parts, TB_RDMA_DEFAULT_OP_TIMEOUT_MS);
    update_stats_from_rdma();
    if (r != RdmaResult::Ok) { close_unlocked(); return RequestStatus::IoError; }

    std::vector<uint8_t> rx;
    r = rdma_->recv_ctrl(&rx, TB_RDMA_DEFAULT_OP_TIMEOUT_MS);
    update_stats_from_rdma();
    if (r != RdmaResult::Ok) { close_unlocked(); return RequestStatus::IoError; }
    if (rx.size() < sizeof(MsgHeader)) { close_unlocked(); return RequestStatus::ProtocolError; }
    MsgHeader rh = {}; std::memcpy(&rh, rx.data(), sizeof(rh));
    if (rh.cmd == CMD_ERROR) {
        if (rx.size() < sizeof(MsgHeader) + sizeof(MsgError)) { close_unlocked(); return RequestStatus::ProtocolError; }
        MsgError ep = {}; std::memcpy(&ep, rx.data() + sizeof(MsgHeader), sizeof(ep));
        if (peer_err_code)   *peer_err_code = ep.code;
        if (peer_err_detail) peer_err_detail->assign(rx.begin() + sizeof(MsgHeader) + sizeof(MsgError), rx.end());
        close_unlocked();
        return RequestStatus::PeerError;
    }
    if (rh.cmd != static_cast<uint8_t>(expected_rsp_cmd) ||
        rh.payload_size != rsp_size + body_size ||
        rx.size() != sizeof(MsgHeader) + rsp_size) {
        close_unlocked();
        return RequestStatus::ProtocolError;
    }
    if (rsp_size) std::memcpy(rsp, rx.data() + sizeof(MsgHeader), rsp_size);
    if (body_size) {
        r = rdma_->recv_data(body_dest, body_size, TB_RDMA_DEFAULT_OP_TIMEOUT_MS);
        update_stats_from_rdma();
        if (r != RdmaResult::Ok) { close_unlocked(); return RequestStatus::IoError; }
    }
    return RequestStatus::Ok;
}

RequestStatus Transport::send_response_with_body(TbRdmaCmd cmd,
                                                 const void * rsp_ctrl, size_t rsp_ctrl_size,
                                                 const void * body, size_t body_size) {
    std::lock_guard<std::mutex> lock(mu_);
    if (mode_ == TransportMode::Tcp) {
        std::vector<Slice> parts = {
            { rsp_ctrl, rsp_ctrl_size },
            { body,     body_size },
        };
        return send_frame_parts(cmd, parts);
    }
    if (!rdma_ || !rdma_->alive()) return RequestStatus::Closed;
    MsgHeader h = {};
    h.cmd          = static_cast<uint8_t>(cmd);
    h.payload_size = static_cast<uint32_t>(rsp_ctrl_size + body_size);
    std::vector<RdmaConnection::Slice> sl = {
        { &h, sizeof(h) },
        { rsp_ctrl, rsp_ctrl_size },
    };
    RdmaResult r = rdma_->send_ctrl_parts(sl, TB_RDMA_DEFAULT_OP_TIMEOUT_MS);
    update_stats_from_rdma();
    if (r != RdmaResult::Ok) { close_unlocked(); return RequestStatus::IoError; }
    if (body_size) {
        r = rdma_->send_data(body, body_size, TB_RDMA_DEFAULT_OP_TIMEOUT_MS);
        update_stats_from_rdma();
        if (r != RdmaResult::Ok) { close_unlocked(); return RequestStatus::IoError; }
    }
    return RequestStatus::Ok;
}

RequestStatus Transport::recv_bulk_body(void * dest, size_t size, const void * inline_src) {
    std::lock_guard<std::mutex> lock(mu_);
    if (size == 0) return RequestStatus::Ok;
    if (mode_ == TransportMode::Tcp) {
        if (!inline_src) { close_unlocked(); return RequestStatus::ProtocolError; }
        std::memcpy(dest, inline_src, size);
        return RequestStatus::Ok;
    }
    if (!rdma_ || !rdma_->alive()) return RequestStatus::Closed;
    RdmaResult r = rdma_->recv_data(dest, size, TB_RDMA_DEFAULT_OP_TIMEOUT_MS);
    update_stats_from_rdma();
    if (r != RdmaResult::Ok) { close_unlocked(); return RequestStatus::IoError; }
    return RequestStatus::Ok;
}

} // namespace ggml_tb_rdma

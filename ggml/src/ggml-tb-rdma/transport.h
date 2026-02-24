#pragma once

// Thunderbolt-RDMA framed transport over a single TCP socket.
//
// Until the M2 milestones land, *all* traffic uses TCP. The framing helpers
// here are designed so that swapping the underlying byte stream to RDMA
// SEND/RECV in M2 only requires changing the `raw_*` primitives — the
// `request()` helper, mutex serialization, and fail-closed semantics
// remain unchanged.

#include "endpoint.h"
#include "protocol.h"
#include "socket.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ggml_tb_rdma { class RdmaConnection; }

namespace ggml_tb_rdma {

// Outcome of `Transport::request()` / `Transport::request_var()`. Anything
// other than `Ok` closes the transport — there is no soft-fail in this
// backend.
enum class RequestStatus {
    Ok,
    PeerError,    // server replied with CMD_ERROR; see `error_code` / `error_detail`
    Disconnected, // peer closed the connection cleanly
    ProtocolError,// framing / cmd / size mismatch, etc.
    IoError,      // OS-level send/recv error
    Closed,       // local side has been close()'d
};

// Hello-time outcome. `Rdma` is reserved for M2a (currently we always end up
// in `Tcp`).
enum class TransportMode {
    Tcp,
    Rdma,
};

// Internal counters — exposed for tests and for `GGML_TB_RDMA_LOG_LEVEL=debug`
// output.
struct TransportStats {
    uint64_t frames_sent     = 0;
    uint64_t frames_received = 0;
    uint64_t bytes_sent      = 0;
    uint64_t bytes_received  = 0;
    // RDMA-only (zero in TCP mode).
    uint64_t rdma_frames_sent     = 0;
    uint64_t rdma_frames_received = 0;
    uint64_t rdma_retransmits     = 0;
    uint64_t rdma_credit_stalls   = 0;
    uint64_t rdma_qp_cq_errors    = 0;
    uint64_t rdma_flow_frames     = 0;
    uint64_t fallback_events      = 0;
};

class Transport {
public:
    Transport();
    ~Transport();

    Transport(const Transport &)             = delete;
    Transport & operator=(const Transport &) = delete;

    // ---- Bring-up -------------------------------------------------------

    // Client side: connect + hello negotiation. `rdma_device` is currently
    // unused (M2a). Returns false and sets *err on any failure (including
    // protocol-version / required-feature mismatch).
    bool client_connect(const Endpoint & ep,
                        const std::string & rdma_device,
                        std::string * err = nullptr);

    // Server side: take ownership of an already-accepted socket and run the
    // hello exchange.
    bool server_accept(int accepted_fd,
                       const std::string & rdma_device,
                       std::string * err = nullptr);

    // ---- Teardown -------------------------------------------------------

    // Sends CMD_GOODBYE on a best-effort basis, then closes the underlying
    // socket. Safe to call multiple times.
    void close();

    bool connected() const { return fd_ >= 0; }
    TransportMode mode() const { return mode_; }
    const TransportStats & stats() const { return stats_; }

    // ---- Request/response helpers --------------------------------------

    // Fixed-size request → fixed-size response.
    //
    // Send a single frame {cmd, sizeof(Req)} carrying `req`, then expect a
    // single response frame {expected_rsp_cmd, sizeof(Rsp)} into `rsp`. If
    // the response cmd is CMD_ERROR, `*peer_err_code` / `*peer_err_detail`
    // are populated (when non-null) and the transport is closed.
    //
    // Any framing/cmd/size mismatch closes the transport and returns
    // ProtocolError — there is deliberately no "drain and continue" path.
    template <typename Req, typename Rsp>
    RequestStatus request(TbRdmaCmd cmd,
                          const Req & req,
                          TbRdmaCmd expected_rsp_cmd,
                          Rsp & rsp,
                          uint16_t * peer_err_code = nullptr,
                          std::string * peer_err_detail = nullptr) {
        return request_raw(cmd, &req, sizeof(Req),
                           expected_rsp_cmd, &rsp, sizeof(Rsp),
                           peer_err_code, peer_err_detail);
    }

    // Empty request → fixed-size response.
    template <typename Rsp>
    RequestStatus request_no_payload(TbRdmaCmd cmd,
                                     TbRdmaCmd expected_rsp_cmd,
                                     Rsp & rsp,
                                     uint16_t * peer_err_code = nullptr,
                                     std::string * peer_err_detail = nullptr) {
        return request_raw(cmd, nullptr, 0,
                           expected_rsp_cmd, &rsp, sizeof(Rsp),
                           peer_err_code, peer_err_detail);
    }

    // Variable-size request → variable-size response.
    //
    // Sends one frame whose payload is the concatenation of `req_parts`
    // (header + tail buffers — useful for graph_compute and set_tensor).
    // The response is a single frame whose payload is read into `*rsp_out`
    // (resized to the actual size). `max_rsp_bytes` caps the response size;
    // a larger frame is treated as ProtocolError.
    struct Slice {
        const void * data;
        size_t       size;
    };
    RequestStatus request_var(TbRdmaCmd cmd,
                              const std::vector<Slice> & req_parts,
                              TbRdmaCmd expected_rsp_cmd,
                              std::vector<uint8_t> * rsp_out,
                              uint32_t max_rsp_bytes,
                              uint16_t * peer_err_code = nullptr,
                              std::string * peer_err_detail = nullptr);

    // ---- Bulk request/response (M2c streaming) -------------------------
    //
    // Send a request whose body should stream as DATA when in RDMA mode.
    // In TCP mode the body is appended to the same frame as `req_ctrl`.
    // The response is fixed-size (no body).
    RequestStatus request_bulk_send(TbRdmaCmd cmd,
                                    const void * req_ctrl, size_t req_ctrl_size,
                                    const void * body, size_t body_size,
                                    TbRdmaCmd expected_rsp_cmd,
                                    void * rsp, size_t rsp_size,
                                    uint16_t * peer_err_code = nullptr,
                                    std::string * peer_err_detail = nullptr);

    // Send a request and receive a response whose body should stream as
    // DATA when in RDMA mode. In TCP mode the body is read from the same
    // frame as the response struct.
    RequestStatus request_bulk_recv(TbRdmaCmd cmd,
                                    const void * req_ctrl, size_t req_ctrl_size,
                                    TbRdmaCmd expected_rsp_cmd,
                                    void * rsp, size_t rsp_size,
                                    void * body_dest, size_t body_size,
                                    uint16_t * peer_err_code = nullptr,
                                    std::string * peer_err_detail = nullptr);

    // ---- Server-side primitives ----------------------------------------

    // Receive the next request frame on this connection. Header is returned
    // in `*hdr_out`. The payload (if non-empty) is read into `*payload_out`
    // (resized to `hdr.payload_size`). `max_payload_bytes` caps the payload
    // size; a larger frame is treated as ProtocolError.
    RequestStatus recv_request(MsgHeader * hdr_out,
                               std::vector<uint8_t> * payload_out,
                               uint32_t max_payload_bytes);

    // Send a fixed-size reply.
    template <typename Rsp>
    RequestStatus send_response(TbRdmaCmd cmd, const Rsp & rsp) {
        return send_response_raw(cmd, &rsp, sizeof(Rsp));
    }

    // Send an empty reply (header only).
    RequestStatus send_response_empty(TbRdmaCmd cmd) {
        return send_response_raw(cmd, nullptr, 0);
    }

    // Send a multi-part reply (variable size). All parts are concatenated
    // into a single frame.
    RequestStatus send_response_var(TbRdmaCmd cmd, const std::vector<Slice> & parts);

    // Server-side: send a response carrying a small `rsp_ctrl` struct plus
    // a streaming body. In TCP mode `body` is concatenated into the same
    // frame; in RDMA mode the body streams as a separate DATA message.
    RequestStatus send_response_with_body(TbRdmaCmd cmd,
                                          const void * rsp_ctrl, size_t rsp_ctrl_size,
                                          const void * body, size_t body_size);

    // Server-side: after `recv_request` returned a header for a cmd with
    // a streaming body, read `size` body bytes into `dest`. In TCP mode
    // the body is already inside the payload that `recv_request` returned;
    // pass `inline_src` (a pointer into that payload) so we can memcpy.
    // In RDMA mode, `inline_src` is ignored and we recv_data().
    RequestStatus recv_bulk_body(void * dest, size_t size,
                                 const void * inline_src);

    // Send a CMD_ERROR frame and close the transport.
    void send_error_and_close(TbRdmaError code, const std::string & detail);

private:
    RequestStatus request_raw(TbRdmaCmd cmd,
                              const void * req, size_t req_size,
                              TbRdmaCmd expected_rsp_cmd,
                              void * rsp, size_t rsp_size,
                              uint16_t * peer_err_code,
                              std::string * peer_err_detail);

    RequestStatus send_response_raw(TbRdmaCmd cmd, const void * payload, size_t size);

    RequestStatus send_frame_raw(TbRdmaCmd cmd, const void * payload, size_t size);
    RequestStatus send_frame_parts(TbRdmaCmd cmd, const std::vector<Slice> & parts);

    RequestStatus recv_frame_header(MsgHeader * hdr_out);
    RequestStatus recv_frame_payload(void * buf, size_t size);

    bool do_hello_exchange_client(std::string * err);
    bool do_hello_exchange_server(std::string * err);

    // M2a: attempt RDMA bring-up after the TCP hello. Returns true if the
    // session is now in RDMA mode (TCP closed). Returns false if we stayed
    // in TCP mode (either because RDMA is unavailable / disabled, or
    // bring-up failed — see logs for why). Sets *err only on a hard error
    // (e.g. GGML_TB_RDMA_REQUIRE_RDMA=1 and RDMA bring-up failed).
    bool try_promote_to_rdma_client(std::string * err);
    bool try_promote_to_rdma_server(std::string * err);

    // RDMA-mode framing (used when mode_ == Rdma). These map a TCP-shaped
    // request/response onto CTRL messages over RDMA SEND/RECV.
    RequestStatus rdma_request_raw(TbRdmaCmd cmd,
                                   const void * req, size_t req_size,
                                   TbRdmaCmd expected_rsp_cmd,
                                   void * rsp, size_t rsp_size,
                                   uint16_t * peer_err_code,
                                   std::string * peer_err_detail);
    RequestStatus rdma_request_var(TbRdmaCmd cmd,
                                   const std::vector<Slice> & req_parts,
                                   TbRdmaCmd expected_rsp_cmd,
                                   std::vector<uint8_t> * rsp_out,
                                   uint32_t max_rsp_bytes,
                                   uint16_t * peer_err_code,
                                   std::string * peer_err_detail);
    RequestStatus rdma_recv_request(MsgHeader * hdr_out,
                                    std::vector<uint8_t> * payload_out,
                                    uint32_t max_payload_bytes);
    RequestStatus rdma_send_response_raw(TbRdmaCmd cmd, const void * payload, size_t size);
    RequestStatus rdma_send_response_var(TbRdmaCmd cmd, const std::vector<Slice> & parts);

    void update_stats_from_rdma();

    void close_unlocked();

    int            fd_   = -1;
    TransportMode  mode_ = TransportMode::Tcp;
    TransportStats stats_{};
    bool           closed_after_error_ = false;

    std::unique_ptr<RdmaConnection> rdma_;
    bool                            is_server_ = false;

    // One in-flight op per connection — kills entire classes of races.
    std::mutex mu_;
};

} // namespace ggml_tb_rdma

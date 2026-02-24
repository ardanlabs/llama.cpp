#pragma once

#include "ggml-backend.h"

#include <string>

namespace ggml_tb_rdma {

// Run a single-client server on `endpoint_str`. Returns when the listening
// socket is closed (SIGINT/SIGTERM via the process-level handler the caller
// has installed) or on unrecoverable bring-up failure.
void run_server(const std::string & endpoint_str,
                const std::string & rdma_device,
                ggml_backend_t      backend);

} // namespace ggml_tb_rdma

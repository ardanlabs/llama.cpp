message(STATUS "Building TB-RDMA backend (macOS / Thunderbolt)")

# This directory is only entered on Darwin (see ../CMakeLists.txt). It is
# safe for sources here to assume __APPLE__.

set(GGML_TB_RDMA_SOURCES
    backend.cpp
    endpoint.cpp
    ibv.cpp
    rdma.cpp
    server.cpp
    socket.cpp
    transport.cpp
)

ggml_add_backend_library(ggml-tb-rdma ${GGML_TB_RDMA_SOURCES})

target_compile_features(ggml-tb-rdma PRIVATE cxx_std_17)

# librdma.dylib is opened at runtime via dlopen() in M2a — no link-time
# dependency required. The header `<infiniband/verbs.h>` ships with the
# macOS SDK (it is required at compile time so the inline post_send/
# post_recv/poll_cq helpers can dispatch through `qp->context->ops`).
target_link_libraries(ggml-tb-rdma PRIVATE ${CMAKE_DL_LIBS})
target_link_libraries(ggml-tb-rdma PRIVATE "-lc++")

# Optional: enable build-time fault injection used by some unit tests.
if (GGML_TB_RDMA_FAULT_INJECT)
    target_compile_definitions(ggml-tb-rdma PRIVATE GGML_TB_RDMA_FAULT_INJECT)
endif()

# -----------------------------------------------------------------------------
# Unit tests (M0.8). Always built when the backend is built — `enable_testing`
# is called unconditionally in the parent ggml CMakeLists.
# -----------------------------------------------------------------------------

find_package(Threads REQUIRED)

add_executable(test-tb-rdma
    tests/test-tb-rdma.cpp
    endpoint.cpp
    ibv.cpp
    rdma.cpp
    socket.cpp
    transport.cpp
)
target_include_directories(test-tb-rdma PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_compile_features(test-tb-rdma PRIVATE cxx_std_17)
target_link_libraries(test-tb-rdma PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
if (GGML_TB_RDMA_FAULT_INJECT)
    target_compile_definitions(test-tb-rdma PRIVATE GGML_TB_RDMA_FAULT_INJECT)
endif()
add_test(NAME test-tb-rdma COMMAND test-tb-rdma)

# -----------------------------------------------------------------------------
# M5: real-hardware tests. Always built so CI can pick them up, but the
# binary self-skips unless GGML_TB_RDMA_RUN_HW_TESTS=1 is set in the env.
# Linked against the actual backend library and ggml-cpu (used by the
# in-process server thread for graph_compute).
# -----------------------------------------------------------------------------

add_executable(test-tb-rdma-hw tests/test-tb-rdma-hw.cpp)
target_include_directories(test-tb-rdma-hw PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_compile_features(test-tb-rdma-hw PRIVATE cxx_std_17)
target_link_libraries(test-tb-rdma-hw PRIVATE ggml ggml-base ggml-tb-rdma Threads::Threads)
add_test(NAME test-tb-rdma-hw COMMAND test-tb-rdma-hw)


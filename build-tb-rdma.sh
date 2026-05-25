#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# build-tb-rdma.sh
#
# Build only the Thunderbolt-RDMA backend (`ggml-tb-rdma`) as a dynamically
# loadable ggml backend, plus the matching unit / hardware test binaries.
#
# This script:
#   1. configures CMake with the minimum set of options needed to produce a
#      loadable `libggml-tb-rdma.dylib` (BUILD_SHARED_LIBS + GGML_BACKEND_DL),
#   2. builds *only* the backend target and its tests (not the rest of the
#      llama.cpp tools / examples),
#   3. runs the unit test,
#   4. prints the artifact paths and step-by-step instructions for placing the
#      dylibs into Kronk's user-managed libraries directory so Kronk loads
#      this build instead of its prebuilt release.
#
# Target platform: macOS (APPLE-only backend).
# -----------------------------------------------------------------------------

set -euo pipefail

# ---- knobs (override via env) ------------------------------------------------
BUILD_DIR="${BUILD_DIR:-build-tb-rdma}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

# Where Kronk loads user-managed llama.cpp libs from when invoked with
# --lib-path (or kronk.WithLibPath). Override KRONK_LIB_PATH if you keep
# yours somewhere else.
KRONK_LIB_PATH="${KRONK_LIB_PATH:-$HOME/.kronk/libraries-tb-rdma}"

# ---- sanity ------------------------------------------------------------------
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "error: ggml-tb-rdma is macOS-only (uname -s = $(uname -s))" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Configuring CMake in ./$BUILD_DIR"
cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_BACKEND_DL=ON \
    -DGGML_NATIVE=ON \
    -DGGML_METAL=ON \
    -DGGML_BLAS=OFF \
    -DGGML_TB_RDMA=ON \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_SERVER=OFF

echo
echo "==> Building ggml-tb-rdma (+ tests) with $JOBS jobs"
cmake --build "$BUILD_DIR" -j "$JOBS" --target \
    ggml-tb-rdma \
    test-tb-rdma \
    test-tb-rdma-hw

echo
echo "==> Running unit test"
ctest --test-dir "$BUILD_DIR" -R '^test-tb-rdma$' --output-on-failure

# ---- locate the artifact -----------------------------------------------------
BIN_DIR="$SCRIPT_DIR/$BUILD_DIR/bin"
DYLIB="$BIN_DIR/libggml-tb-rdma.dylib"

if [[ ! -f "$DYLIB" ]]; then
    # Fall back to a search in case the layout changes upstream.
    DYLIB="$(find "$SCRIPT_DIR/$BUILD_DIR" -name 'libggml-tb-rdma.dylib' -print -quit || true)"
fi

if [[ -z "${DYLIB:-}" || ! -f "$DYLIB" ]]; then
    echo "error: libggml-tb-rdma.dylib was not produced under $BUILD_DIR" >&2
    exit 1
fi

# Collect every dylib emitted by this build so Kronk gets a self-contained
# set (libllama, libggml, libggml-base, libggml-cpu, libggml-metal,
# libggml-blas if enabled, libmtmd if it was built, and libggml-tb-rdma).
mapfile -t ALL_DYLIBS < <(find "$BIN_DIR" -maxdepth 1 -name '*.dylib' | sort)

echo
echo "============================================================"
echo " Build complete."
echo "============================================================"
echo
echo "Backend dylib:"
echo "  $DYLIB"
echo
echo "All built dylibs (copy these together):"
for f in "${ALL_DYLIBS[@]}"; do
    echo "  $f"
done

# ---- post-build placement instructions ---------------------------------------
cat <<EOF

------------------------------------------------------------
 Use this build from Kronk
------------------------------------------------------------

Kronk loads llama.cpp dylibs from a single directory. Per
sdk/tools/libs/libs.go, an existing directory WITHOUT a
version.json file is treated as a user-managed read-only
build, which is what we want here.

1. Create the target directory and copy every dylib from
   this build into it (do NOT create version.json):

     mkdir -p "$KRONK_LIB_PATH"
     cp $BIN_DIR/*.dylib "$KRONK_LIB_PATH/"

2. Quick sanity-check that the backend dylib has no missing
   link-time dependencies:

     otool -L "$KRONK_LIB_PATH/libggml-tb-rdma.dylib"

3. Point Kronk at this directory when starting it. Either:

     # via env var picked up by Kronk's --lib-path flag /
     # libs.WithLibPath option (whichever your kronk binary
     # exposes — see 'kronk --help' / 'kronk server --help'):
     kronk server --lib-path "$KRONK_LIB_PATH"

     # or by overriding the libraries root before launch:
     KRONK_LIB_PATH="$KRONK_LIB_PATH" make kronk-server

4. Confirm the backend registered at startup. The ggml log
   line you want to see is:

     load_backend: loaded TB-RDMA backend from $KRONK_LIB_PATH/libggml-tb-rdma.dylib

   (Kronk's --lib-path adds the directory to LD_LIBRARY_PATH
   and calls ggml_backend_load_all on it.)

Note: this only makes the backend *available*. Selecting it
for a model still requires the Kronk CLI flags being added
in the follow-up plan ('--rpc' and '--tb-rdma').
EOF

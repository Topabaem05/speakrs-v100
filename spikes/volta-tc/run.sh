#!/usr/bin/env bash
# Spike A for the speakrs V100 port: can cuda-oxide's ptx_asm! carry Volta tensor-core
# instructions to working sm_70 code?
#
# Builds the kernels for sm_70 three ways and runs each through the host, once as PTX
# (driver JIT) and once as an sm_70 cubin from the CUDA 12.9 ptxas:
#   plain    wmma spelled for PTX ISA 6.0, the version cuda-oxide emits for sm_70
#   aligned  wmma with `.aligned`, the module raised to PTX 6.5 by an llc wrapper
#   m8n8k4   the aligned build plus the mma.sync m8n8k4 kernel, which needs PTX 6.4
# A failing step is reported and the next one still runs.
#
# Run inside the speakrs-v100 dev container: `bash /src/spikes/volta-tc/run.sh`
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${SPIKE_OUT:-/cache/target/volta-tc}"
mkdir -p "$out"

# cuda-oxide passes `-mattr=+ptxNN` to llc only for instructions it detects, and it
# detects neither wmma nor m8n8k4, so this wrapper raises the whole module instead
wrapper="$out/llc-ptx65"
cat >"$wrapper" <<EOF
#!/usr/bin/env bash
exec "${CUDA_OXIDE_LLC:-/usr/bin/llc-21}" "\$@" -mattr=+ptx65
EOF
chmod +x "$wrapper"

# an explicit target directory, since the dev box sets CARGO_TARGET_DIR for everything
cargo build -q --release --manifest-path "$here/host/Cargo.toml" --target-dir "$out/host" ||
    { echo "host build failed"; exit 1; }
host="$out/host/release/volta-tc-host"

status=0
build_and_run() {
    local name="$1" features="$2" llc="$3"
    local dir="$out/$name" target="$out/target-$name"
    rm -rf "$dir" && mkdir -p "$dir"
    echo "=== $name (features: ${features:-none})"
    if ! (cd "$here/kernels" &&
        cargo clean -q -p volta-tc-kernels --target-dir "$target" 2>/dev/null
        CUDA_OXIDE_PTX_DIR="$dir" CUDA_OXIDE_LLC="$llc" CARGO_TARGET_DIR="$target" \
            cargo oxide build --arch sm_70 ${features:+--features "$features"}); then
        echo "$name: cuda-oxide build FAILED"
        status=1
        return
    fi
    local ptx="$dir/volta_tc_kernels.ptx"
    grep -m2 -E '^\.(version|target)' "$ptx"
    echo "--- PTX through the driver JIT"
    "$host" "$ptx" || status=1
    echo "--- sm_70 cubin from $("$CUDA12_HOME/bin/ptxas" --version | grep '^Cuda')"
    if "$CUDA12_HOME/bin/ptxas" -arch=sm_70 "$ptx" -o "$dir/volta_tc_kernels.sm_70.cubin"; then
        "$host" "$dir/volta_tc_kernels.sm_70.cubin" || status=1
    else
        echo "$name: ptxas FAILED"
        status=1
    fi
}

build_and_run plain "" "${CUDA_OXIDE_LLC:-/usr/bin/llc-21}"
build_and_run aligned "aligned" "$wrapper"
build_and_run m8n8k4 "aligned,m8n8k4" "$wrapper"
exit "$status"

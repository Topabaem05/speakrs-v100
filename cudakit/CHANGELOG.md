# Changelog

## Unreleased

Volta (V100, `sm_70`) as an opt-in tier below the `sm75` baseline.

### cudakit

- `PtxTier::Sm70`, and `Profile::floor`: the lowest tier a build accepts devices for.
  It is the baseline unless the build compiles `Sm70` in, so builds without it still
  refuse compute capability 7.0 exactly as before
- `Error::UnsupportedDevice` names the build's floor instead of always `sm_75`

### cudakit-build

- `Tier::Sm70`: when a project targets it, every area ships an `sm70` variant that
  exports exactly the baseline's kernels, gets only `sm_70` cubins, and is linted
  against sm_75+ instructions (`ptx_lint::check_sm70_opcodes`)
- **Breaking:** `Pins::ptxas_legacy`, the CUDA 12 ptxas (`CUDA12_HOME`) that builds
  cubins below sm_75, which CUDA 13 dropped. Manifests record it as `ptxas.sm_70`;
  manifests without such cubins are unchanged
- `scripts/setup-gpu-box.sh` takes `CUDA12_DOTTED` and exports `CUDA12_HOME`

## 0.1.0

First release.

### cudakit

- `Gpu` and `Runtime<K>`: one context and stream per device, with a module cache for
  an application's kernel areas that loads exact cubins or PTX by identity and never
  substitutes another artifact. Both are `Send + Sync`, and cudakit makes the context
  current on whichever thread calls it
- `Profile` and `LibraryPolicy`: build policy checked at open, including driver-only
  operation that refuses every library before anything is looked up
- `PtxTier`, `TierSet` and `AreaPtx` leave room for new tiers: `PtxTier::ALL` is a
  slice, `AreaPtx` is built with `EMPTY`, `baseline` and `with`, and a profile names
  features only for the tiers it ships
- cuBLAS and cuDNN loaded at run time after a symbol preflight (feature `libraries`)
- `Sgemm` and `SgemmBatched`: row-major products with length checks, and
  `unsafe fn Gpu::blas` for other cuBLAS calls on the shared handle
- `Gpu::capture` and `Graph`: graph capture that pauses cudarc's event tracking and
  detects empty captures
- `dnn`: a deterministic cuDNN convolution planner with fused bias, residual and ReLU,
  and `Conv2d::check` for convolution geometry
- `DeviceTensor`, `unfold_windows` and `identity::DriverRelease`

### cudakit-build

- `Project` with `build`, `build_cubins` and `check` for cuda-oxide kernel crates,
  pinned cubins, per-area manifests and generated PTX tier crates
- `Project::targets`: the tiers the consumer has features for, so a tier added to
  cudakit-build later is not required of existing consumers
- `ptx_lint`, which rejects generic shared-memory addresses narrowed without a
  conversion to the shared space

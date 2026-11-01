# speakrs V100(sm_70) CUDA 커널 추가 계획

## Context

- 블로그(avencera, "Writing GPU kernels in Rust: speakrs 60x faster than pyannote")의 speakrs는 Rust 화자 분리 라이브러리다.
  - 커널: NVIDIA cuda-oxide(`#[kernel]` Rust → PTX)로 직접 쓴다.
  - 호스트: cudakit(cudarc 기반, 드라이버만 필요)으로 실행한다.
  - 배포: PTX와 cubin을 오프라인으로 빌드해 crate에 커밋한다.
  - 검증과 측정: RTTM/DER로 정확도를 확인하고, A/B 교대 측정으로 성능을 잰다.
  - 결과: RTX 4090에서 pyannote 대비 60배.
- speakrs는 sm_75(Turing) 이상만 지원한다(tier sm75/sm80/sm120). **V100(sm_70)은 런타임을 열 때 `UnsupportedDevice`로 거부되고 cuDNN 폴백도 없다.**
- 목표: 블로그의 방식 그대로 speakrs와 cudakit fork에 `Sm70` tier를 추가해, domoba-gpu의 V100에서 native CUDA backend를 실행한다.
  1. 정확도: DER이 CPU backend와 같다(차이 ≤0.01%p).
  2. 성능: pyannote(V100), speakrs CPU, cuDNN/cuBLAS 후보보다 빠르다. 참고로 블로그의 다른 호스트 측정값은 T4 294×, A100 1296× RTFx다.
  3. 메모리: 공유 GPU에서 VRAM ≤4GB.
- 사용자 결정(2026-11-01): speakrs fork 방식, GPU는 **항상 공유 상태**.
- 갱신(2026-11-01, M1 검증 후): **GPU를 전부 써도 된다.** VRAM 4GB 예산과 공유 대응 규칙은 더 이상 제약이 아니다. 다만 다른 프로세스(llama-server 2개, python 2개)가 약 11GB를 계속 점유할 수 있으므로, 측정 때마다 `nvidia-smi`로 상태를 기록한다.

## 대상 환경 (2026-11-01 SSH 실측, `domoba-gpu` = 100.73.38.99)

| 항목 | 값 |
|---|---|
| GPU | Tesla V100-SXM2-16GB ×1, driver 580.178.04, ECC on, persistence on, compute mode Default |
| GPU 공유 | llama-server 2개와 python 2개가 약 10.9GB 사용 중 → 여유 약 5GB |
| CPU/RAM | Ryzen 3 3200G 4코어/4스레드, RAM 29GB(가용 약 12GB) |
| CUDA | 시스템 nvcc/ptxas 12.0. pip(torch 2.10.0+cu128)에 cuDNN 9.10.2.21, cuBLAS 12.8.4.1, NVRTC 12.8.93 |
| 없음 | Rust, ncu/nsys, LLVM 21, 암호 없는 sudo. `RmProfilingAdminOnly: 1` |
| 있음 | docker 29.1.3과 `nvidia` runtime(docker 그룹), 디스크 여유 `/` 95G, `/mnt/storage` 338G, Ubuntu 24.04.4 |

## V100(sm_70) 제약 (1차 출처 확인)

| 항목 | 사실 | 설계 영향 |
|---|---|---|
| 하드웨어 | 80 SM, FP32 15.7 TF, Tensor FP16 125 TF, 900 GB/s, L2 6MB, shared ≤96KB/block(>48KB opt-in), SM당 2048 thread, 255 reg/thread | T4(40 SM, 64KB) 기준 tile과 grid를 다시 고른다 |
| sm_75+ 명령 | `mma.m16n8k8.f16`, `ldmatrix`, `tanh.approx`, `ex2.approx.f16` | h16 재작성, lint로 차단 |
| sm_80+ 명령 | `cp.async`, TF32/BF16 mma, `redux.sync`, `cvt.rn.f16x2.f32` | 쓰지 않는다. f16 pack은 `cvt.rn.f16.f32` 두 번 |
| Volta Tensor Core | `wmma` f16 m16n16k16(PTX 6.0, `.aligned`는 6.3부터), `mma.sync.m8n8k4`(PTX 6.4) | `ptx_asm!`으로 직접 쓴다. cuda-oxide catalog에는 Volta mma/wmma intrinsic이 없다 |
| cuda-oxide | 문서는 sm_80+. 고정 rev `918bbde`는 sm_70 floor를 PTX 6.0으로 기록. `ldmatrix`가 있으면 sm_70 거부. wmma와 m8n8k4는 감지하지 않음. PTX 철자에 6.4가 없음. `--materialize-cubin`(libNVVM 13.x)은 sm_70 불가 | `--arch sm_70`을 명시하고, llc → PTX → 외부 ptxas 경로만 쓴다 |
| 도구 사슬 | CUDA 13.0이 Volta 오프라인 컴파일을 제거. 마지막 12.x는 12.9 Update 1. ptxas 12.9는 PTX ≤8.8, R580 JIT는 PTX ≤9.0 | sm_70 cubin은 ptxas 12.9로 만든다 |
| 드라이버·라이브러리 | R580이 Volta 마지막 branch(LTSB, 2028-06 종료). cuDNN 9.10.2가 cc 7.0 마지막 지원(9.11에서 제거). cuBLAS는 12.x까지 | 580.x에 고정. 비교용 library는 cuDNN 9.10.2와 cuBLAS 12.x |
| 프로파일러 | Nsight Compute 2025.2.x가 Volta 마지막. Nsight Systems는 2025.4부터 미지원 | 컨테이너에 ncu 2025.2.x와 nsys ≤2025.3 |

## 현재 코드에서 막히는 지점

1. cudakit `crates/cudakit/src/profile.rs`: cc < `BASELINE`(Sm75)이면 `UnsupportedDevice`를 낸다. `PtxTier`(`tier.rs`)에 Sm70이 없다.
2. speakrs `src/inference/cuda/implementation.rs`: 상시 area(Fbank, Embedding, Segmentation)가 `PtxTier::BASELINE`을 하드코딩한다(`qualified_module`, `production_tier`, `AreaTarget::for_area`).
3. cudakit-build(`crates/cudakit-build/src/lib.rs`):
   - tier 목록이 sm75로 시작한다고 가정한다(`Area::new`, `check_entry_points`, `variant_cubins`, `Area::embeds`, `may_add`).
   - ptxas는 13.0.88 하나만 고정한다.
4. 커널: sm_70에서 쓸 수 없는 명령은 `spk_wideconv_h16_*` 41개 entry(`crates/speakrs-cuda-kernels/src/wideconv/h16.rs`, tier gate 없음)에만 있다.
   - `mma.m16n8k8.f16` 8,784회, `ldmatrix` 738회
   - 나머지 sm75 PTX는 PTX 6.0에서 합법인 FP32 SIMT 명령만 쓴다.
5. FP16 선택 규칙이 tier가 아니라 cc 기준이다: `Config::fp16`(cc == 7.5), `Oxide::tuning_fp16_pin`(capability < TURING).
6. 테스트가 cc 7.0 거부를 단언한다: `src/inference/cuda/tier.rs`, `tests/runtime.rs:137`. `tuning/tests.rs`는 cc.major == 7을 Turing으로 본다.

## 설계 결정

- **tier:** `PtxTier::Sm70`을 Sm75 아래에 추가하고 `BASELINE = Sm75`는 유지한다. 판정은 `Profile::floor()`(빌드에 포함된 최저 tier)로 한다. 그래서 sm70을 넣지 않은 빌드는 지금처럼 cc 7.0을 거부한다.
- **feature:** `cuda-sm70`(별칭 `cuda-v100`). 단독이면 driver-only다. `cuda`와 함께 켜면 cuDNN/cuBLAS 폴백이 붙는다. M2 검증 전에는 `cuda` 묶음에 넣지 않는다.
- **산출물:** 새 crate `crates/speakrs-ptx-sm70`. PTX와 sm_70 cubin 하나만 담고, 압축 크기는 2.5~3.5MB로 예상한다.
- **ptxas 이중 고정:** 기존 tier는 13.0.88(`CUDA13_HOME`), sm_70 cubin은 12.9(`CUDA12_HOME`, `Pins.ptxas_legacy`).
- **cuda-oxide는 rev `918bbde`로 유지한다.** 버전을 올리지 않는다.
- **단계 구분:** M1은 FP32 SIMT만 쓴다(h16은 trap stub과 tier guard). M2는 `wmma` m16n16k16 Tensor Core로 h16을 대체한다. `mma.m8n8k4`는 프로파일이 요구할 때만 쓴다.

## 작업 단계

### Phase 0. 준비와 스파이크 (go/no-go)

1. **저장소:**
   - 이 계획을 `GPU_Rust/PLAN-v100.md`로 저장한다.
   - upstream speakrs와 cudakit을 `GPU_Rust/speakrs`, `GPU_Rust/cudakit`에 clone하고 branch `v100-sm70`을 만든다.
   - speakrs는 `[patch.crates-io]`로 로컬 cudakit과 cudakit-build를 쓴다.
   - GitHub fork 생성과 push는 사용자 확인 후에 한다.
2. **서버:**
   - Mac에서 편집하고 `domoba-gpu:/mnt/storage/gpu-rust/`로 rsync한다.
   - `target/`, 모델, 데이터셋(VoxConverse test 첫 10개, dev 216개)은 `/mnt/storage`에 둔다.
   - MPS daemon 유무를 확인한다.
3. **컨테이너 `docker/v100-dev.Dockerfile`:**
   - `ubuntu:24.04`에서 cudakit fork의 `scripts/setup-gpu-box.sh`를 실행한다. 이 스크립트가 LLVM 21, `cuda-nvcc-13-0`, `libnvvm-13-0`, rustup stable과 `nightly-2026-08-28`, cargo-oxide `918bbde`를 설치한다.
   - 이어서 speakrs의 `scripts/cuda/setup-gpu-box.sh`로 `/workspace` 배치와 모델을 준비한다.
   - fork에서 `cuda-nvcc-12-9`(ptxas 12.9, `CUDA12_HOME`), ncu 2025.2.x, nsys ≤2025.3을 추가한다.
   - 기존 규칙(`/workspace/env.sh`, `PINS`, `gpu-bench.lock` flock)을 재사용한다.
   - 실행: `docker run --runtime nvidia --gpus all --cap-add SYS_ADMIN -v /mnt/storage/gpu-rust:/workspace ...`. `cargo oxide doctor`가 통과하는지 확인한다.
4. **스파이크 A (도구 사슬):**
   - `probe`, `fbank`, `sincnet`을 `cargo oxide build --arch sm_70`으로 빌드한다. `.target sm_70` 확인 → ptxas 12.9로 cubin 생성 → 임시 패치한 cudakit으로 V100에서 load → ORT CPU reference 비교.
   - wmma 실험 커널로 확인할 것:
     - PTX 6.0에서 `.aligned` 없는 철자가 ptxas를 통과하는지
     - 32-bit shared 주소를 쓸 수 있는지
     - 안 되면 `CUDA_OXIDE_LLC` wrapper(`llc-21 "$@" -mattr=+ptx65`)와 `.aligned` 철자로 되는지
   - m8n8k4 실험 커널은 wrapper와 함께 빌드해 C fragment 배치를 확인한다.
   - 서버의 `$CUDA12_HOME/bin/ptxas --version` 줄을 그대로 기록한다(고정값으로 쓴다).
   - **Go 기준:** 세 area가 허용 오차를 통과하고, wmma(또는 wrapper 방식)가 16×16×16 GEMM 정답을 낸다.
   - **No-go 대응:** 실패 원인이 cuda-oxide에 있으면 해당 부분만 cuda-oxide fork에 패치하고 rev를 고정한다. upstream issue도 올린다.
5. **스파이크 B (기준선):**
   - pyannote community-1: `~/PYTHON_ENV_GUIDE.md` 규칙의 system-site venv(torch 공유)와 `scripts/pyannote-bench/diarize.py`
   - speakrs CPU backend: 4스레드
   - 같은 파일 세트로 DER과 RTF를 `benchmarks/` 형식에 맞춰 기록한다.

### Phase 1. cudakit 변경 (cudakit 0.1.1, cudakit-build 0.2.0)

- **`crates/cudakit/src/tier.rs`:** `Sm70`을 첫 variant로 추가한다(파생 `Ord`가 tier 비교에 쓰이기 때문). name "sm70", min (7,0)로 하고 `index`, `ALL`, `FromStr`를 갱신한다.
- **`profile.rs`:** `pub fn floor()`를 추가하고 `select()`의 BASELINE 비교를 floor로 바꾼다. `Profile`에 field를 더하지 않아 호환이 깨지지 않는다.
- **`error.rs`:** "(sm_75, Turing)" 하드코딩을 tier 정보로 바꾼다. cc 7.0인데 sm70이 빌드에 없으면 `cuda-v100`을 켜라고 안내한다.
- **`crates/cudakit-build/src/lib.rs`:**
  - `Tier::Sm70`: arch "sm_70", capability 70, feature `tier-sm70`, `min_ptx_isa` (6,0)
  - `Area::new`가 `[Sm70, Sm75, …]`를 허용한다.
  - `Area::embeds`의 pinned 조건에 `tier >= BASELINE`을 더한다. `may_add`는 Sm70에서 false다.
  - `check_entry_points`의 기준은 BASELINE 모듈이다. Sm70 entry는 이것과 완전히 같아야 한다.
  - `Project::variant_cubins`: BASELINE 아래 tier는 `arch < 75`만 포함한다(sm70 → [70]).
  - `Pins.ptxas_legacy: Option<&'static str>`를 추가한다. `Toolchain { current, legacy }::for_arch()`를 `build`, `build_cubins`, `build_area_cubins`, `rebuild_cubins`에 적용한다.
  - `Project::validate`: 75 미만 arch에는 `ptxas_legacy`가 있어야 한다. Sm70 target이면 모든 area가 Sm70을 가져야 한다.
- **`manifest.rs`:** 선택적 헤더 `ptxas.sm_70 = …`를 추가한다. 기존 manifest는 byte 단위로 그대로다.
- **ptx lint:** `check_sm70_opcodes`를 추가한다. ldmatrix, movmatrix, `mma.sync.aligned.m16n8`, cp.async, tanh.approx, ex2.approx.f16, cvt.rna.tf32, redux를 거부한다. cuda-oxide가 m16n8k8을 감지하지 못하므로 이것이 유일한 조기 검사다.
- **테스트:**
  - sm70 미포함 빌드에서 7.0은 계속 거부된다.
  - {Sm70, Sm75} 빌드에서 7.0 → Sm70, 7.5 → Sm75, 6.1 → 거부.
  - sm70 → [70] cubin.
  - `ptxas.sm_70` manifest를 저장했다가 다시 읽어도 같아야 하고, 줄이 틀리거나 빠지거나 남으면 거부한다.
  - sm70 entry가 더해지거나 빠지면 실패한다.
  - lint가 ldmatrix와 m16n8k8을 잡는다.

### Phase 2. speakrs M1: FP32 경로로 V100 실행

1. `xtask/src/commands/cuda_kernels.rs`의 `PROJECT`: targets에 Sm70을 넣고, `cubin_arches` 앞에 `CubinArch(70)`을 넣는다. `ptxas_legacy`에는 스파이크 A에서 기록한 버전 줄을 넣는다.
2. 커널 crate:
   - `tier-sm70 = []` feature를 추가한다. `lib.rs`에서 `tier-sm80`과 함께 켜면 `compile_error!`를 낸다.
   - `h16.rs`의 `h16_conv3x3!`, `h16_conv3x3_s2!` 본문과 `ldsm4`, `mma` helper를 `not(tier-sm70)`으로 감싼다. sm70에서는 기존 패턴의 `cuda_device::debug::trap()` stub을 넣어 entry 목록을 유지한다.
3. `Cargo.toml`:
   - member와 dependency에 `speakrs-ptx-sm70`을 추가한다.
   - feature: `cuda-sm70 = ["_cuda", "dep:speakrs-ptx-sm70"]`, `cuda-v100 = ["cuda-sm70"]`
   - 함께 갱신할 곳: docs.rs feature 목록, `lib.rs` compile_error 목록, ci.yml과 justfile matrix, package.yml의 `-p` 목록과 크기 검사
4. `crates/speakrs-ptx-sm70/Cargo.toml`은 sm75를 복사해 손으로 쓴다. `src/lib.rs`와 `ptx/`는 생성한다.
5. `src/inference/cuda/kernels.rs`: `shipped_ptx`/`shipped_cubin` cfg에 `cuda-sm70`을 넣는다. 모든 area에 `.with(PtxTier::Sm70, tier_ptx!(["cuda-sm70"], "ptx/<area>.sm70", [70]))`을 추가한다.
6. `src/inference/cuda/tier.rs`: `compiled.with_if(Sm70, cfg!(feature = "cuda-sm70"))`, `SHIPPED = [Sm70, Sm75]`, `tier_features`에 `(Sm70, "cuda-sm70")`을 추가한다.
7. `implementation.rs`: `always_on_tier(device, limit)`을 추가한다. device < 7.5이거나 limit < Sm75이면 Sm70을 돌려준다. `qualified_module`, `production_tier`, `AreaTarget::for_area`에서 쓴다.
8. FP16 guard:
   - `src/inference/cuda/candidate/wideconv.rs`의 `Oxide::with_config`와 DeviceRule 경로에서 tier < Sm75이면 `Algorithm::Fp16(_)`을 거부한다.
   - `tuning_fp16_pin`은 tier로 판정한다.
9. 문구와 문서: "Turing (7.5) or newer"가 들어간 곳(`error.rs`, `runtime.rs`, `docs/cuda.md`, README, `scripts/cuda/README.md`, cudakit README)과 CHANGELOG를 고친다.
10. GPU box에서 전체 산출물을 다시 만든다. kernel crate의 sources hash가 바뀌어 모든 manifest가 stale이 되기 때문이다. 확인할 것:
    - sm75/80/120의 PTX와 cubin이 byte 단위로 같은지
    - production binding이 고정한 hash가 그대로인지: wideconv sm120 cubin `414f3c83…`, lstm/sincnet sm75 PTX `72945743…`/`967bc689…`
- **V100에서의 동작:** binding 없이 sm70 cubin과 FP32 trunk를 쓴다. C32/C64는 direct resnet, C128/C256은 FFMA Winograd로 간다. FP32 Winograd(62,464B)는 기존 >48KB opt-in 경로로 동작한다(V100 한도 98,304B).
- **테스트 갱신:**
  - `tier.rs`, `tests/runtime.rs:137`, `kernels.rs`의 cubin 선택 테스트: 기대값을 `cuda-sm70` 여부에 따라 나눈다.
  - `kernel_inventory_tests`: device 7.0을 추가한다.
  - golden: "Tesla V100-SXM2-16GB"(7.0, 80 SM) 행을 추가한다.
  - `driver/tests.rs`: opt-in 표에 (7.0, Sm70, 98304)를 넣고, FP16이 선택되지 않는지 확인한다.
  - 신규: 상시 area가 7.0에서 Sm70을 고르는지, Sm70에서 Fp16 설정이 거부되는지 확인한다.
- **M1 완료 조건:**
  - `cuda-v100` 빌드가 V100에서 reference test와 `tests/cuda_pipeline.rs`를 통과한다.
  - library가 적재되지 않았음을 증명한다.
  - VoxConverse test 10개의 DER이 CPU backend와 같다.

### Phase 3. speakrs M2: Volta Tensor Core FP16 trunk

- **현재 h16 구조:**
  - block은 128 thread다. warp마다 출력 1행 × 64열을 맡고, MT개 m16 tile × 8개 n8 tile을 계산한다(lane당 FP32 누적 64개(c32) 또는 128개(c64)).
  - staging: 8채널/16B pixel-major double buffer. A fragment는 global ring buffer, B는 `ldmatrix.x4` 2회로 읽는다.
  - epilogue: 레지스터에서 ×2^-20 → residual → bias → ReLU(NaN 유지). half 출력은 warp별 shared tile을 거친다. 포화 시 `red.global.or`로 range flag를 켠다.
- **wmma 방식 (권장):** 같은 entry, 인자, grid를 쓰는 `tier-sm70` 본문을 만든다.
  - 명령: `wmma.load.a…global.f16`(출력 8), `wmma.load.b…shared.f16`(출력 8), `wmma.mma.sync.row.col.m16n16k16.f32.f32`(누적 8, 입력 16 = `ptx_asm!` 입력 한도), `wmma.store.d…shared.f32`. 철자(`.aligned` 유무, wrapper)는 스파이크 A 결과로 정한다.
  - shared: 16채널/32B pixel-major double buffer. stride 1은 25,344B, stride 2는 43,520B로 48KB 미만이다.
    - host `Shape::fp16_launch`가 tier를 받아 Sm70이면 32B/pixel로 계산한다.
    - staging 저장 순서를 조정해 quarter-warp bank conflict를 없앤다.
  - 가중치: `spk_wideconv_pack_h16`의 sm70 본문이 row-major 16×16 f16 tile `[cin/16][tap][cout/16][16][16]`을 쓴다.
  - epilogue: n-tile 4개를 warp별 `[16][68]` f32 tile(padding 포함)에 저장 → `syncwarp` → coalesced 저장. 스케일, residual, bias, ReLU, range flag는 같다.
  - 레지스터: c64에서 lane당 약 200개. spill이 나면 sm70에서 `launch_bounds(128,1)`을 쓴다.
- **m8n8k4 (조건부):** ncu에서 wmma load/store가 병목으로 보일 때만 시도한다. PTX 6.4가 필요하므로 llc wrapper나 upstream 감지 규칙이 있어야 한다.
- **numerics:** 피연산자 반올림(×2^10, `cvt.rn`, 포화)과 epilogue가 같다. MMA 내부의 합산 순서만 다르므로 T4 허용 오차를 그대로 쓴다.
- **테스트:**
  - 41개 entry 전부 T4 허용 오차 이내
  - 포화 입력 시 range flag와 재계산
  - `_half` 출력 체인이 FP32 쌍과 bit 단위로 일치
  - dev 216개 DER
- `VoltaFp16Trunk`(cc == 7.0) recipe는 DER 불변과 tune 이득이 확인된 뒤에만 추가한다.
- **M2 완료 조건:** DER 변화 ≤0.01%p. trunk GPU 시간의 M1 대비 변화를 측정해 보고한다.

### Phase 4. V100 튜닝, VRAM, CPU 병목 (공유 GPU)

- **tune:** `speakrs cuda tune`으로 sm70 tune file을 만든다.
  - 후보: FP32 direct/Winograd, h16 sm70, library(`cuda` 결합 빌드)
  - 공유 GPU 대응: round를 7에서 15로 늘린다. 두 번 실행해 같은 선택만 채택하고, 5% 미만 차이는 "차이 없음"으로 본다.
- **tile과 launch:** 80 SM과 96KB shared 기준으로 다시 고른다(segdense, resnet, wideconv, lstm).
- **VRAM:**
  - speakrs에는 VRAM 상한 옵션이 없고, 저장소가 batch-32 class까지 커진다(layer1 tensor 하나가 327MB).
  - batch class(1/4/8/16/32)별 최대 사용량을 먼저 잰다.
  - 4GB를 넘으면 `RuntimeConfig`에 `EmbeddingBatchClass` 상한을 추가하고, 시작 시 `cuMemGetInfo`로 상한을 정한다.
- **cooperative LSTM:** time-slicing 아래에서도 결과는 맞지만 지연이 흔들린다. 분산을 함께 보고한다.
- **4코어 CPU:** stage별 시간(디코드, seg GPU, powerset, emb GPU, PLDA/AHC/VBx)과 GPU-only 처리량을 따로 보고한다. CPU stage를 GPU로 옮기는 일은 범위 밖 후속 과제다.

### Phase 5. M3: 패키징, CI, 문서

- `crates/speakrs-ptx-sm70`의 크기를 확인한다(10 MiB 제한).
- `cargo xtask cuda-kernels check`(toolkit 없는 CI 작업 포함)와 GPU box의 `check --rebuild`(ptxas 두 개)가 통과해야 한다.
- `scripts/cuda/prove-driver-only.sh`를 `cuda-v100` 빌드로 V100에서 실행한다.
- `docs/cuda.md`에 V100 tier, driver 580.x 고정, 빌드용 ptxas 12.9, VRAM 주의 사항을 쓴다.
- M2 검증 후 `cuda` 묶음에 포함할지, upstream PR(speakrs, cudakit)을 낼지는 사용자가 정한다.

## 검증

1. **후보 테스트 (V100, 컨테이너):** 기존 허용 오차를 그대로 쓴다.
   - seg logits 2e-3
   - fbank 2e-3(ORT), 1e-3(f64)
   - segdense ≤1e-3
   - embedding cosine ≥0.99999(FP32), ≥0.999(FP16/TF32 모드)
   - resnet/wideconv/lstmproj: cuDNN 9.10.2 결과와 f64 참값
2. **틀린 경우 검출:**
   - epilogue scale을 일부러 틀린 h16 sm70 빌드는 실패해야 한다.
   - 포화 입력을 주입하면 재계산 결과가 FP32와 정확히 같아야 한다(`fp16_range_tests`).
   - sm70이 없는 빌드는 cc 7.0을 계속 거부해야 한다.
   - lint는 sm_75 명령이 든 sm70 PTX를 거부해야 한다.
3. **호스트 테스트:** `cargo xtask cuda-kernels check`, golden/inventory/loader 테스트를 Mac이나 CI에서 돌린다.
4. **E2E 정확도:** VoxConverse test 10개와 dev 216개로 DER(`src/metrics.rs`, collar 0)을 잰다. CPU backend 대비 차이 ≤0.01%p. FP32 경로는 RTTM byte-identical을 목표로 한다.
5. **성능:** stage별 GPU 시간(CUDA event)과 speakrs-bm RTFx를 잰다.
   - 방법: `gpu-bench.lock` 아래 A/B 교대 라운드. 중앙값과 IQR을 보고하고, 라운드마다 `nvidia-smi --query-compute-apps`를 기록한다.
   - 비교 대상: pyannote, speakrs CPU, library 후보, sm70 FP32, sm70 FP16
6. **메모리:** batch class별 최대 VRAM ≤4GB.

## 위험과 대응

| 위험 | 대응 |
|---|---|
| ptxas 12.9 버전 줄과 byte 재현성(pip, docker, installer 간 차이) | 서버 컨테이너에서 읽은 값으로 고정하고 `check --rebuild`로 확인한다. |
| wmma 철자(`.aligned`, 32-bit shared 주소)가 확인되지 않음 | 스파이크 A에서 probe PTX를 먼저 assemble한다. |
| cuda-oxide가 감지하지 못하는 sm_75 명령 | `check_sm70_opcodes` lint로 막는다. |
| trap stub이 호출되면 CUDA context가 망가짐 | tier guard와 테스트로 sm70의 FP16 선택을 막는다. |
| 전체 산출물 재생성 | 기존 tier의 byte 동일성과 binding hash를 확인한다. |
| 공유 GPU의 VRAM 변동과 타이밍 노이즈 | VRAM 상한, 교대 측정, 중앙값으로 대응한다. 긴 벤치마크는 다른 작업이 적은 시간에 돌린다. |
| ncu 관리자 권한 | 컨테이너 `--cap-add SYS_ADMIN`을 쓴다. 안 되면 CUDA event와 `ptxas -v`(register/spill/smem)로 대신한다. |
| R580 이후 Volta 지원 종료(2028-06) | driver 580.x에 고정하고 PTX ≤9.0을 유지한다. 문서에 위험을 적는다. |
| 4코어 CPU 병목 | GPU-only 지표를 따로 보고한다. |

## 재사용할 기존 코드

- trap stub 패턴 `#[cfg(not(feature = "tier-sm80"))] { cuda_device::debug::trap(); }`
- cp.async가 없는 staging: `wideconv.rs`의 `stage_scalar` → `sts1`
- FP16 range flag(`red.global.or`)와 `Fp16Policy::Excluded` 재계산, 가중치 기준 FP16 제외 규칙
- tune harness(`tuning/bench.rs`, `tuning/accuracy.rs`), `scripts/cuda/make_reference.py`, `export_weights.py`, `prove-driver-only.sh`
- cudakit과 speakrs의 `setup-gpu-box.sh`, `src/metrics.rs`, speakrs-bm, `scripts/pyannote-bench/diarize.py`

## 완료 후 기록

- 검증된 재사용 지식을 llm-wiki 공통 루트에 트랜잭션으로 기록하고, Hindsight index를 sync한다. 대상: sm_70 도구 사슬, cuda-oxide의 sm_70 동작, 서버 제약.
- 프로젝트 전용 기록은 사용자가 프로젝트 루트를 고른 경우에만 남긴다.

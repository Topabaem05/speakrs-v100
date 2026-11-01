# speakrs V100(sm_70) 작업 현황과 남은 작업

기준일: 2026-11-02. 계획 문서는 [PLAN-v100.md](PLAN-v100.md)다.

## 1. 구성

| 위치 | 내용 |
|---|---|
| `speakrs/` | avencera/speakrs fork, branch `v100-sm70`, 기준 커밋 `669e810`. 변경은 아직 커밋하지 않았다(44개 파일) |
| `cudakit/` | cudakit fork, branch `v100-sm70`. `Sm70` tier, `Profile::floor()`, ptxas 이중 고정(CUDA 13.0 + 12.9) |
| `spikes/volta-tc/` | Volta `wmma`/`m8n8k4` 검증 스파이크 |
| `tools/voxconverse_hf/` | VoxConverse dev를 HF 미러에서 받아 설치하는 도구 |
| 서버 `domoba-gpu:/mnt/storage/gpu-rust/` | 소스 사본(`src/`), 빌드(`target/`), 모델(`models/`), 데이터(`datasets/`), 배포(`deploy/`) |
| 서버 컨테이너 `speakrs-v100` | `docker/v100-dev.Dockerfile`로 만든 개발 컨테이너(CUDA 12.9, cuDNN 9.10.2, ncu 2025.2.1) |

## 2. 완료한 것

- **M1:** 모든 kernel area에 `sm70` variant를 추가했다. `cuda-v100`(`cuda-sm70`) feature로 V100을 받아들인다. FP32 경로의 DER은 CPU, T4와 같다.
- **M2:** wideconv FP16 trunk tile을 Volta `wmma` m16n16k16으로 다시 썼다(stride 1, stride 2, half I/O 변형 42개 entry).
  - tap 루프에 `.pragma "nounroll"`을 넣어 spill을 없앴다.
  - V100의 TF32 모드(embedding 기본값)가 stem을 뺀 모든 3×3 trunk 층에서 FP16 tile을 쓰도록 라우팅했다(`DeviceDefault::VoltaFp16Trunk`).
- **CI와 패키징:** package.yml, ci.yml, justfile에 sm70 crate와 feature를 넣었다. sm70 crate는 압축 후 2.46MB다.
- **검증:**
  - V100 GPU 테스트: lib 370개, cuda_pipeline 4개가 통과한다.
  - 층별 정확도 검사 410건이 통과한다.
  - 틀린 epilogue scale은 검사에서 실패한다(음성 대조).

## 3. 성능 (빈 GPU, 4코어 호스트)

| 측정 | M1 (FP32 trunk) | M2 (FP16 trunk) |
|---|---|---|
| dev 216 DER, `cuda` | 7.012545% | 7.011792% (RTTM 213/216 동일) |
| dev 216 DER, `cuda-fast` | 7.367643% | 7.367643% (RTTM 215/216 동일) |
| 긴 파일 4개 A/B 중앙값, `cuda` | 16.8초 | 5.6초 (3.00×) |
| 긴 파일 4개 A/B 중앙값, `cuda-fast` | 9.9초 | 3.2초 (3.09×) |
| dev 216 하네스 RTFx | 측정 조건 다름 | `cuda` 752×, `cuda-fast` 1,293× |
| + V100 tune 파일 | | `cuda` 766×, `cuda-fast` 1,312×, DER·RTTM 변화 없음 |

- 한 번만 돌린 전체 실행 시간은 실행마다 약 15% 흔들린다. 대표값은 번갈아 잰 A/B 배율이다.
- 공유 GPU에서 잰 값(RTFx 56.7)은 time-slicing 때문에 약 3배 부풀려졌다. 비교에 쓰지 않는다.

## 4. 서버 배포 상태

- **실행:** `/mnt/storage/gpu-rust/deploy/speakrs-v100 [--mode cuda|cuda-fast] WAV... > out.rttm`. 호스트에서 바로 실행된다.
- **구성 요소:**
  - `deploy/bin/speakrs-v100-m2`: M2 빌드(`cuda,cuda-v100`). SHA-256은 `e0f47b90d2266589…`로 시작한다.
  - `deploy/cuda-tune-v100.json`: V100 tune 파일. SHA-256은 `1f8f9a94ad2205a9…`로 시작한다. 바꾸는 것은 stem batch 32의 Spatial 선택뿐이다.
- **확인:** `grzbb.wav`(607초)의 RTTM이 컨테이너의 M2 결과와 같다. 로그에 `Loaded CUDA tune file`이 찍힌다.
- **tune 파일 제약:** 이 파일은 장치, driver(580.178.04), speakrs 버전과 내장 커널 hash가 맞을 때만 쓰인다. driver나 빌드가 바뀌면 경고를 내고 무시하며, 그때는 `speakrs cuda tune`으로 다시 만든다.
- **멈춰 둔 작업:** 측정을 위해 llama-server 2개, GPU 모니터, domoba 학습 daemon을 멈췄다.
  - daemon에는 `runs/PAUSE`도 걸려 있다.
  - 학습 ar026, ar027은 평가 전에 중지했고, `ar_state.json`에는 pending으로 남아 있다.
  - 복원은 `target/scripts/restore-gpu-users.sh`로 한다. 2026-11-02 현재 사용자 요청으로 복원하지 않은 상태다.

## 5. 남은 작업

### 5.1 trunk FP16 커널 효율 (효과 추정 10~25%, 작업 큼)

batch 32 운영 커널을 Nsight Compute로 분석했다(SM 1.30GHz 고정).

| 커널 | 시간 | tensor pipe | 레지스터 | 점유율 |
|---|---|---|---|---|
| `h16_c64_half_in_out` | 868µs | 52.6% | 227 | 12.5% (SM당 8 warp) |
| `h16_c128_half_in_out` | 767µs | 59.6% | 226 | 12.5% |

- **가장 큰 stall:** `wait`(발행 명령당 1.41~1.47 cycle)이고, 그다음이 `long_scoreboard`(약 1.0)다.
- **SASS 관찰:** `wmma` 하나가 HMMA.884 묶음 4개로 풀리는데, 같은 누산기에 쌓는 묶음이 연달아 나온다(STEP0~3 × k 조각 4개). 이 의존 사슬이 `wait`의 유력한 원인이다. 아직 가설이다.
- **shared store bank conflict:** wavefront의 9~13%가 conflict이고, ncu는 최대 8.8~12.9% 개선을 추정한다. 원인은 8채널씩 반쪽 단위로 staging할 때 32B 픽셀 간격에서 생기는 2-way conflict다.
- **후보 작업:**
  1. `mma.sync.aligned.m8n8k4`(PTX 6.4)로 바꾸고, k 조각을 여러 출력 tile에 번갈아 배치해 의존 사슬을 끊는다. cuda-oxide가 PTX 6.0을 쓰므로 llc wrapper(`-mattr=+ptx65`)가 필요하다. 스파이크는 `spikes/volta-tc`에 있다.
  2. half 입력 변형에서는 16채널을 한 번에 staging한다. lane을 (픽셀 4개 × 반쪽 2개)로 섞으면 store conflict가 없어진다. 레지스터는 +16개 정도 늘고 255 이내다.
  3. A(가중치) fragment를 다음 tap 것까지 미리 읽어 `long_scoreboard`를 줄인다.
- **검증 절차:** `trunk-check.sh`(층별 정확도), `trunk-timing.sh`(batch별 시간), ncu, A/B, dev 216 DER을 차례로 돌린다.

### 5.2 LSTM (효과 추정 전체 3~7%, 작업 큼)

- 운영 batch(1, 32)에서 현재 plan 선택이 이미 최적이다. batch 32에서 tiled 7.77ms, wide 9.02ms다.
- 우리 LSTM은 cuDNN보다 1.8~3.8배 빠르다.
- batch 32 tiled 커널은 step마다 약 3.3µs를 L2 hidden-state 교환에 쓴다.
- 줄이려면 V100 전용 커널이 필요하다. 방안은 두 가지다.
  - recurrent 행렬을 2개 block으로 나누는 커널
  - 행렬을 FP16으로 보관하는 커널. DER 검증이 필요하다.

### 5.3 작은 항목

- **stem 규칙:** Volta에서는 batch 32에서도 stem을 Spatial로 둔다(0.912ms → 0.470ms, 전체 약 1~2%). 지금은 tune 파일로만 적용되어 있고, 코드 규칙으로 넣으면 tune 파일이 필요 없다.
- **batch별 VRAM 측정:** M1의 최대 사용량은 2,358MiB였다. M2는 아직 재지 않았다.
- **`#[unroll]` 경고:** cuda-oxide가 32회 × 3,206 연산 루프의 펼침을 거부한다는 경고를 낸다. 모든 tier에서 같아 upstream 쪽 사안이며, 원인 위치는 확인하지 않았다.

### 5.4 정리와 공개

- speakrs, cudakit fork의 변경을 커밋하고, GitHub(`Topabaem05/V100_fast_` 등)에 push할지 정해야 한다.
- 정할 것이 두 가지 더 있다. `cuda-sm70`을 `cuda` 묶음에 넣을지, upstream(avencera/speakrs, cudakit)에 PR을 낼지다.
- README의 GPU 표는 16코어 호스트에서 10개 파일 subset으로 잰 값이다. V100 행을 넣으려면 같은 조건으로 다시 재야 한다.

## 6. 지식 기록

LLM wiki(revision 745)에 다음을 기록했다. Hindsight 색인에는 sync했다.

- `case-ptxas-unrolls-sm70-wmma-tap-loop-20261101`
- `case-shared-gpu-inflates-v100-timing-20261101`
- `case-ncu-counter-permission-docker-root-20261101`
- `source-speakrs-v100-sm70-fp16-trunk-20261101`

# speakrs-v100

[speakrs](https://github.com/avencera/speakrs)의 native CUDA 화자 분리를 Tesla V100(compute capability 7.0, sm_70)에서 돌리기 위한 fork 모음이다. upstream speakrs는 sm_75(Turing) 이상만 지원한다.

## 결과 (Tesla V100-SXM2-16GB, 4코어 호스트)

| 측정 | FP32 trunk (M1) | Volta FP16 trunk (M2) |
|---|---|---|
| VoxConverse dev DER, `cuda` | 7.0125% | 7.0118% |
| VoxConverse dev DER, `cuda-fast` | 7.3676% | 7.3676% |
| 처리 속도 (A/B 중앙값) | 기준 | 3.0~3.1배 |
| dev 216 RTFx (benchmark 하네스) | | `cuda` 752×, `cuda-fast` 1,293× |

자세한 현황과 남은 작업은 [V100_STATUS.md](V100_STATUS.md), 처음 계획은 [PLAN-v100.md](PLAN-v100.md)에 있다.

## 구성

| 폴더 | 내용 | upstream 기준 |
|---|---|---|
| `speakrs/` | `cuda-v100`(`cuda-sm70`) feature, sm70 커널 variant, Volta `wmma` FP16 trunk tile, V100 라우팅 | avencera/speakrs `669e810` |
| `cudakit/` | `PtxTier::Sm70`, CUDA 12.9 ptxas 고정, sm70 lint | avencera/cudakit (fork branch `v100-sm70`) |
| `spikes/volta-tc/` | Volta `wmma`/`mma.m8n8k4` 동작 검증 스파이크 | |
| `tools/voxconverse_hf/` | VoxConverse dev를 Hugging Face 미러에서 받아 설치하는 도구 | |

`speakrs/Cargo.toml`은 `[patch.crates-io]`로 옆 폴더 `../cudakit`을 쓴다. 그래서 두 폴더는 같은 위치에 함께 둔다.

## 빌드와 실행

```bash
cd speakrs
cargo build --release -p xtask --features cuda,cuda-v100
target/release/xtask diarize --mode cuda --models-dir <models> input.wav > out.rttm
```

`cuda-v100`만 켜면 cuDNN, cuBLAS 없이 driver만으로 동작한다. 함께 넣은 PTX와 cubin을 쓰므로 일반 빌드에는 CUDA toolkit이 필요 없다. 커널을 다시 만들 때 필요한 도구 사슬과 컨테이너는 [speakrs/docs/cuda.md](speakrs/docs/cuda.md)의 Volta 절과 `speakrs/docker/v100-dev.Dockerfile`을 참고한다. sm_70 cubin에는 CUDA 12.9 ptxas가 필요하다(CUDA 13이 Volta를 제거했다).

## 라이선스

speakrs와 cudakit은 각 폴더의 Apache-2.0 라이선스를 따른다.

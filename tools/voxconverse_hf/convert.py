# /// script
# requires-python = ">=3.12"
# dependencies = ["pyarrow"]
# ///
"""Build a speakrs-style VoxConverse dev directory (wav/ and rttm/) from the Hugging Face
mirror diarizers-community/voxconverse, for when the Oxford VGG server that hosts the
original WAV archive is unreachable

The mirror's audio column keeps each clip's file name and WAV bytes. This writes those
bytes unchanged and pairs each clip with the RTTM of the same name from the official
repository, https://github.com/joonson/voxconverse. It refuses a row that is not 16 kHz
mono 16-bit WAV, a name without an RTTM, a duplicate name, or a file count other than
the expected one, and prints the total duration to compare with the published 1217.8
minutes of VoxConverse dev
"""

import argparse
import io
import json
import re
import shutil
import sys
import wave
from pathlib import Path

import pyarrow.parquet as pq

VOXCONVERSE_ID = re.compile(r"^[a-z]{5}$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument(
        "--rttm-dir", type=Path, required=True, help="the dev/ directory of the official repository"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="writes <output>/wav and <output>/rttm"
    )
    parser.add_argument("--expect", type=int, default=216)
    parser.add_argument("--report", type=Path, help="write a JSON summary here")
    args = parser.parse_args()

    wav_dir = args.output / "wav"
    rttm_dir = args.output / "rttm"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rttm_dir.mkdir(parents=True, exist_ok=True)

    problems: list[str] = []
    seen: dict[str, float] = {}
    for parquet in sorted(args.parquet_dir.glob("dev-*.parquet")):
        reader = pq.ParquetFile(parquet)
        for batch in reader.iter_batches(batch_size=4, columns=["audio"]):
            for audio in batch.column("audio").to_pylist():
                data, path = audio.get("bytes"), audio.get("path") or ""
                stem = Path(path).stem
                if not VOXCONVERSE_ID.match(stem):
                    problems.append(f"{parquet.name}: audio path {path!r} is not a VoxConverse id")
                    continue
                if stem in seen:
                    problems.append(f"{stem}: appears more than once")
                    continue
                if not data or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
                    problems.append(f"{stem}: audio bytes are not a WAV file")
                    continue
                with wave.open(io.BytesIO(data)) as clip:
                    params = (clip.getnchannels(), clip.getsampwidth(), clip.getframerate())
                    seconds = clip.getnframes() / clip.getframerate()
                if params != (1, 2, 16000):
                    problems.append(f"{stem}: (channels, bytes, rate) {params}, expected (1, 2, 16000)")
                    continue
                rttm = args.rttm_dir / f"{stem}.rttm"
                if not rttm.is_file():
                    problems.append(f"{stem}: no RTTM in {args.rttm_dir}")
                    continue
                (wav_dir / f"{stem}.wav").write_bytes(data)
                shutil.copyfile(rttm, rttm_dir / f"{stem}.rttm")
                seen[stem] = seconds

    if len(seen) != args.expect:
        problems.append(f"paired {len(seen)} files, expected {args.expect}")
    unused = sorted({p.stem for p in args.rttm_dir.glob("*.rttm")} - set(seen))
    if unused:
        problems.append(f"{len(unused)} RTTMs have no audio: {unused[:5]}")

    minutes = sum(seen.values()) / 60
    summary = {"files": len(seen), "minutes": round(minutes, 1), "problems": problems}
    print(json.dumps(summary, indent=2))
    if args.report:
        args.report.write_text(json.dumps(summary, indent=2) + "\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env bash
# Fetch VoxConverse dev from the Hugging Face mirror while the Oxford VGG server is down,
# pair it with the official RTTMs, and install it as speakrs's `voxconverse-dev`
#
# Run inside the speakrs-v100 dev container: `bash /src/tools/voxconverse_hf/fetch.sh`
# Sources, all public:
#   audio  https://huggingface.co/datasets/diarizers-community/voxconverse (CC-BY-4.0), pinned
#   RTTMs  https://github.com/joonson/voxconverse (dev/)
# The installed manifest's provenance names the Oxford archive, because speakrs records
# its catalog source; SOURCE.json beside the cache records where these files came from
set -euo pipefail

repo="diarizers-community/voxconverse"
revision="3acfa1b45ca4b7419aee999d67d94c617f9c9d47"
datasets="${SPEAKRS_DATASETS_DIR:-/data}"
cache="$datasets/.voxconverse-hf"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$cache/parquet"

echo "=== parquet files at $repo@$revision"
curl -fsSL --retry 3 "https://huggingface.co/api/datasets/$repo/revision/$revision?blobs=true" \
    >"$cache/api.json"
python3 - "$cache/api.json" >"$cache/expected.sha256" <<'EOF'
import json, sys
for sibling in json.load(open(sys.argv[1]))["siblings"]:
    name = sibling["rfilename"]
    if name.startswith("data/dev-"):
        print(f"{sibling['lfs']['sha256']}  {name.removeprefix('data/')}")
EOF
cut -d' ' -f3 "$cache/expected.sha256" | while read -r file; do
    if [[ ! -s "$cache/parquet/$file" ]]; then
        curl -fL --retry 5 -C - -o "$cache/parquet/$file.part" \
            "https://huggingface.co/datasets/$repo/resolve/$revision/data/$file"
        mv "$cache/parquet/$file.part" "$cache/parquet/$file"
    fi
done
(cd "$cache/parquet" && sha256sum -c "$cache/expected.sha256")

echo "=== official RTTMs"
if [[ ! -d "$cache/github/.git" ]]; then
    git clone --quiet --depth 1 https://github.com/joonson/voxconverse "$cache/github"
fi
rttm_commit="$(git -C "$cache/github" rev-parse HEAD)"

echo "=== pair audio and RTTMs into $datasets/voxconverse"
uv run --quiet --no-project "$here/convert.py" --parquet-dir "$cache/parquet" \
    --rttm-dir "$cache/github/dev" --output "$datasets/voxconverse" --report "$cache/convert.json"

python3 - "$cache" "$repo" "$revision" "$rttm_commit" <<'EOF'
import json, sys
cache, repo, revision, rttm_commit = sys.argv[1:]
summary = json.load(open(f"{cache}/convert.json"))
json.dump({
    "audio": f"https://huggingface.co/datasets/{repo}@{revision}",
    "rttm": f"https://github.com/joonson/voxconverse@{rttm_commit} (dev/)",
    "files": summary["files"],
    "minutes": summary["minutes"],
}, open(f"{cache}/SOURCE.json", "w"), indent=2)
print(open(f"{cache}/SOURCE.json").read())
EOF

echo "=== install as voxconverse-dev"
/cache/target/release/xtask dataset ensure voxconverse-dev --datasets-dir "$datasets"
ls "$datasets/voxconverse-dev"

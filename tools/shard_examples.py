"""Split the large per-atom example JSONs in data/label into gzip shards small enough to commit to GitHub.

The originals (data/label/*_examples.json) live under the gitignored 3.1 TB
data/ tree of the working repository, so they are not reachable from a clean
checkout. This script writes them out as
reports/label_examples/<name>.part{i:02d}.json.gz.

Run: python3 tools/shard_examples.py
Reassemble:
    import gzip, json, glob
    d = {}
    for p in sorted(glob.glob("reports/label_examples/grad_v2_examples.part*.json.gz")):
        with gzip.open(p, "rt") as f:
            d.update(json.load(f))
"""

import gzip
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC_DIR = REPO / "data" / "label"
OUT_DIR = REPO / "reports" / "label_examples"

# (filename, number of shards). Target is about 15-20 MB per gzipped shard,
# comfortably below GitHub's 100 MB per-file limit.
TARGETS: list[tuple[str, int]] = [
    ("grad_v2_examples.json", 8),
    ("act_v2_full_examples.json", 16),
]


def shard_file(name: str, n_shards: int) -> None:
    src = SRC_DIR / name
    data: dict[str, dict] = json.loads(src.read_text())
    keys = sorted(data.keys(), key=int)
    stem = name.removesuffix(".json")
    per = (len(keys) + n_shards - 1) // n_shards
    for i in range(n_shards):
        chunk = {k: data[k] for k in keys[i * per : (i + 1) * per]}
        out = OUT_DIR / f"{stem}.part{i:02d}.json.gz"
        with gzip.open(out, "wt") as f:
            json.dump(chunk, f)
        print(f"{out.name}: atoms {len(chunk)}, {out.stat().st_size / 1e6:.1f}MB")


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    for name, n in TARGETS:
        shard_file(name, n)

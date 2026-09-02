# Per-atom example JSONs, as gzip shards

Copies of the large per-atom example files, split so they fit in a git repository.
The originals live in the gitignored `data/label/` tree of the working repository
(`grad_v2_examples.json` 511 MB, `act_v2_full_examples.json` 1.36 GB) and are not
part of this snapshot. Generating script: `code/shard_examples.py`.

- `grad_v2_examples.part00-07.json.gz`: top activating examples for all 32,768 atoms of the grad_v2 dictionary
- `act_v2_full_examples.part00-15.json.gz`: top activating examples for all 32,768 atoms of the act_v2 dictionary

Structure: `{atom_id: {mass, fires, examples: [{row, val, strength, src, text}, ...]}}`.
Shards split the atom ids in sorted order into equal parts (4,096 atoms per shard for
grad, 2,048 for act).

Reassembling, or reading a single atom:

```python
import gzip, json, glob

# merge everything
d = {}
for p in sorted(glob.glob("reports/label_examples/grad_v2_examples.part*.json.gz")):
    with gzip.open(p, "rt") as f:
        d.update(json.load(f))

# a single atom (for example 3057). grad has 4,096 atoms per shard, so part = 3057 // 4096 = 0
with gzip.open("reports/label_examples/grad_v2_examples.part00.json.gz", "rt") as f:
    atom = json.load(f)["3057"]
```

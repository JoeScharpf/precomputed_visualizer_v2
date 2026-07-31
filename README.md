# Precomputed Visualizer — 2 anchors

Same static looping demo as `precomputed_visualizer_v2`, but packs are baked with
**HiPrune α = 0.15** so selection uses **2 anchors** (plus buffers / registers
from the usual formula). Retention stays **25%**.

## Run locally

```bash
npm install
npm run dev
```

## Controls

| Key | Pack |
|-----|------|
| `1` | Parakeet |
| `2` | Surfer |
| `3` | Volcano |
| `4` | Chefchaouen |
| `5` | Signac |
| `6` | Yorkie |
| `7` | Fuji |
| `R` | Restart |
| `Esc` | Idle |

## Tests

```bash
python3 -m pytest tests/test_hiprune_pack_invariants.py -v
```

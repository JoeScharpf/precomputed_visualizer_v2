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

Press a number once to **select** that pack (static original). Press the
**same number again** to **start** the animation. Press it a third time to
freeze back to the static cue (handy for re-takes).

| Key | Pack |
|-----|------|
| `1` | Parakeet |
| `2` | Surfer |
| `3` | Volcano |
| `4` | Chefchaouen |
| `5` | Signac |
| `6` | Yorkie |
| `7` | Fuji |
| `8` | Turtle |
| `9` | Alps |
| `R` | Start / restart playback |
| `Esc` | Idle |

## Tests

```bash
python3 -m pytest tests/test_hiprune_pack_invariants.py -v
```

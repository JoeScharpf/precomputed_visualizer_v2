#!/usr/bin/env python3
"""Precompute demo packs with alpha tuned for ~2 HiPrune anchors."""
from __future__ import annotations

import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:8300/api/infer"
PROMPT = "Describe this image in detail."
# round(budget * alpha / 5) == 2 for typical demo token counts @ 25% retention
ALPHA = 0.15
RETENTION = 0.25
PACKS = (
    "parakeet",
    "surfer",
    "volcano",
    "chefchaouen",
    "signac",
    "yorkie",
    "fuji",
    "turtle",
)
LABELS = {
    "parakeet": "Parakeet",
    "surfer": "Surfer",
    "volcano": "Volcano",
    "chefchaouen": "Chefchaouen",
    "signac": "Signac",
    "yorkie": "Yorkie",
    "fuji": "Fuji",
    "turtle": "Turtle",
}
ROOT = Path(__file__).resolve().parent


def slim_scores(scores: dict | None) -> dict | None:
    if not scores:
        return scores
    keep = {}
    for key in ("object_layer", "deep_layer"):
        if key in scores:
            keep[key] = scores[key]
    return keep or None


def slim_metadata(md: dict | None) -> dict | None:
    if not md:
        return md
    out = {k: v for k, v in md.items() if k != "scores"}
    scores = slim_scores(md.get("scores"))
    if scores:
        out["scores"] = scores
    return out


def infer(image_path: Path) -> dict:
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"
    body = {
        "model": "gemma4",
        "prompt": PROMPT,
        "image": data_url,
        "method": "hiprune",
        "retention": RETENTION,
        "alpha": ALPHA,
        "with_baseline": True,
        "return_vision_attention": True,
        "max_tokens": 256,
    }
    req = urllib.request.Request(
        API,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    wanted = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    packs = [p for p in PACKS if wanted is None or p in wanted]
    if not packs:
        print(f"no packs matched {sorted(wanted or [])}", file=sys.stderr)
        return 1
    for pack_id in packs:
        img = ROOT / pack_id / "image.jpg"
        if not img.exists():
            print(f"missing {img}", file=sys.stderr)
            return 1
        print(f"infer {pack_id} (alpha={ALPHA}) …", flush=True)
        try:
            raw = infer(img)
        except urllib.error.HTTPError as e:
            print(e.read().decode("utf-8", "replace")[:800], file=sys.stderr)
            raise
        result = {
            "answer": raw.get("answer"),
            "baseline_answer": raw.get("baseline_answer"),
            "usage": raw.get("usage"),
            "metadata": slim_metadata(raw.get("metadata")),
        }
        pack = {
            "id": pack_id,
            "label": LABELS.get(pack_id, pack_id.title()),
            "prompt": PROMPT,
            "method": "hiprune",
            "retention": RETENTION,
            "alpha": ALPHA,
            "model": "gemma4",
            "image": f"/demo-packs/{pack_id}/image.jpg",
            "result": result,
        }
        out = ROOT / pack_id / "pack.json"
        out.write_text(json.dumps(pack, indent=2) + "\n")
        md = result.get("metadata") or {}
        scores = (md.get("scores") or {}).get("object_layer") or []
        print(
            f"  anchors={len(md.get('anchors') or [])} "
            f"buffers={len(md.get('buffers') or [])} "
            f"registers={len(md.get('registers') or [])} "
            f"scores={len(scores)} deep={'deep_layer' in (md.get('scores') or {})}",
            flush=True,
        )
        if len(md.get("anchors") or []) != 2:
            print(
                f"  WARNING: expected 2 anchors, got {len(md.get('anchors') or [])}",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

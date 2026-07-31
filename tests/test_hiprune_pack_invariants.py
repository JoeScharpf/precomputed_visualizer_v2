"""HiPrune selection invariants on precomputed demo packs.

For each pack image, check:

1. Middle / object-layer attention: anchors are exactly the top-k
   object-layer scores (HiPrune's shallow selection).
2. Final / deep-layer attention: registers are exactly the top-k
   deep-layer scores among tokens not already kept as anchors/buffers.

These match ``hiprune_select`` in ``vllm/vllm/multimodal/hiprune.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PACKS_ROOT = Path(__file__).resolve().parents[1] / "public" / "demo-packs"
PACK_IDS = (
    "parakeet",
    "surfer",
    "volcano",
    "chefchaouen",
    "signac",
    "yorkie",
    "fuji",
)


def _load_pack(pack_id: str) -> dict:
    path = PACKS_ROOT / pack_id / "pack.json"
    assert path.exists(), f"missing pack {path}"
    return json.loads(path.read_text())


def _topk_indices(scores: list[float], k: int) -> set[int]:
    """Indices of the k largest scores (stable on ties via index order)."""
    assert k >= 0
    if k == 0:
        return set()
    ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return set(ranked[:k])


@pytest.fixture(params=PACK_IDS)
def pack(request) -> dict:
    return _load_pack(request.param)


@pytest.fixture
def metadata(pack: dict) -> dict:
    md = pack["result"]["metadata"]
    assert md.get("method") == "hiprune"
    return md


def test_pack_has_required_categories(metadata: dict, pack: dict) -> None:
    scores = metadata.get("scores") or {}
    assert "object_layer" in scores, f"{pack['id']}: missing scores.object_layer"
    assert "deep_layer" in scores, (
        f"{pack['id']}: missing scores.deep_layer — re-run pack precompute "
        "with deep_layer retained"
    )
    for key in ("anchors", "buffers", "registers", "pruned", "num_tokens", "alpha"):
        assert key in metadata, f"{pack['id']}: missing metadata.{key}"
    n = metadata["num_tokens"]
    assert len(scores["object_layer"]) == n
    assert len(scores["deep_layer"]) == n


def test_anchors_are_top_object_layer_attention(metadata: dict, pack: dict) -> None:
    """Anchors = top-k middle-layer (object_layer) attention scores."""
    scores = metadata["scores"]["object_layer"]
    anchors = list(metadata["anchors"])
    k = len(anchors)
    assert k == 2, f"{pack['id']}: expected 2 anchors (α=0.15 build), got {k}"

    expected = _topk_indices(scores, k)
    actual = set(anchors)
    assert actual == expected, (
        f"{pack['id']}: anchors {sorted(actual)} != top-{k} object_layer "
        f"{sorted(expected)}"
    )

    # Anchors beat every non-anchor on object-layer score (allowing ties
    # only among the top-k set itself).
    min_anchor = min(scores[i] for i in anchors)
    for i, s in enumerate(scores):
        if i in actual:
            continue
        assert s <= min_anchor + 1e-12, (
            f"{pack['id']}: non-anchor {i} score {s} > weakest anchor {min_anchor}"
        )


def test_registers_are_top_deep_layer_attention_outside_shallow(
    metadata: dict, pack: dict
) -> None:
    """Registers = top-k final-layer scores among tokens not anchors/buffers."""
    deep = metadata["scores"]["deep_layer"]
    anchors = set(metadata["anchors"])
    buffers = set(metadata["buffers"])
    registers = list(metadata["registers"])
    shallow = anchors | buffers
    k = len(registers)
    assert k > 0, f"{pack['id']}: expected at least one register"

    eligible = [i for i in range(len(deep)) if i not in shallow]
    # Rank eligible by deep score; take top k.
    ranked = sorted(eligible, key=lambda i: (-deep[i], i))
    expected = set(ranked[:k])
    actual = set(registers)

    assert actual.isdisjoint(shallow), (
        f"{pack['id']}: registers overlap anchors/buffers: "
        f"{sorted(actual & shallow)}"
    )
    assert actual == expected, (
        f"{pack['id']}: registers {sorted(actual)} != top-{k} deep_layer "
        f"outside shallow set {sorted(expected)}"
    )

    min_reg = min(deep[i] for i in registers)
    for i in eligible:
        if i in actual:
            continue
        assert deep[i] <= min_reg + 1e-12, (
            f"{pack['id']}: eligible non-register {i} deep score {deep[i]} "
            f"> weakest register {min_reg}"
        )


def test_buffers_are_spatial_neighbors_of_anchors(metadata: dict, pack: dict) -> None:
    """Buffers are the 4-neighborhood of anchors (clamped), minus anchors."""
    grid_w, _grid_h = metadata["grid"]
    n = metadata["num_tokens"]
    anchors = set(metadata["anchors"])
    buffers = set(metadata["buffers"])

    neighbor_union: set[int] = set()
    for a in anchors:
        for d in (-1, 1, -grid_w, grid_w):
            j = a + d
            if 0 <= j < n:
                neighbor_union.add(j)
    expected_buffers = neighbor_union - anchors
    assert buffers == expected_buffers, (
        f"{pack['id']}: buffers {sorted(buffers)} != "
        f"anchor neighbors {sorted(expected_buffers)}"
    )


def test_kept_partition_covers_all_tokens(metadata: dict, pack: dict) -> None:
    n = metadata["num_tokens"]
    anchors = set(metadata["anchors"])
    buffers = set(metadata["buffers"])
    registers = set(metadata["registers"])
    pruned = set(metadata["pruned"])
    kept = anchors | buffers | registers

    assert not (anchors & buffers), f"{pack['id']}: anchors overlap buffers"
    assert not (anchors & registers), f"{pack['id']}: anchors overlap registers"
    assert not (buffers & registers), f"{pack['id']}: buffers overlap registers"
    assert kept.isdisjoint(pruned), f"{pack['id']}: kept overlaps pruned"
    assert kept | pruned == set(range(n)), (
        f"{pack['id']}: kept∪pruned does not cover [0, {n})"
    )

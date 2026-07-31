import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";
import type { ModelKey, PruningMetadata } from "./lib/types";

const ANCHOR = "rgb(255,60,60)";
const BUFFER = "rgb(255,170,40)";
const REGISTER = "rgb(50,255,80)";
const DIVERSE = "rgb(70,140,255)";
const PROMPT = "rgb(205,90,255)";
const PIVOT = "rgb(255,60,60)";
const UNIFORM = "rgb(45,212,191)";
const PRUNED_FILL = "rgba(0,0,0,0.82)";

/** Map normalized attention in [0,1] to a dark reveal-mask overlay.
 * High attention → transparent (image shows through); low attention →
 * opaque cover. Matches the pruned-overlay tone so both views feel related. */
function heatFill(t: number): string {
  const x = Math.min(1, Math.max(0, t));
  const a = 0.82 * (1 - x);
  return `rgba(12,10,9,${a.toFixed(3)})`;
}

/** Opaque jet-style colormap for the recording demo heatmap beat. */
function heatColorFill(t: number): string {
  const x = Math.min(1, Math.max(0, t));
  let r = 0;
  let g = 0;
  let b = 0;
  if (x < 0.25) {
    const u = x / 0.25;
    r = 0;
    g = Math.round(40 + 215 * u);
    b = 255;
  } else if (x < 0.5) {
    const u = (x - 0.25) / 0.25;
    r = 0;
    g = 255;
    b = Math.round(255 * (1 - u));
  } else if (x < 0.75) {
    const u = (x - 0.5) / 0.25;
    r = Math.round(255 * u);
    g = 255;
    b = 0;
  } else {
    const u = (x - 0.75) / 0.25;
    r = 255;
    g = Math.round(255 * (1 - 0.85 * u));
    b = 0;
  }
  return `rgb(${r},${g},${b})`;
}

function normalizeScores(scores: number[]): number[] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of scores) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const span = hi - lo;
  if (!(span > 0) || !Number.isFinite(span)) {
    return scores.map(() => 0);
  }
  return scores.map((v) => (v - lo) / span);
}

interface Category {
  name: string;
  indices: number[];
  color: string;
  /** Ordered categories report a rank in the tooltip (top-k / pick order). */
  ranked: boolean;
}

/** All kept-token categories in draw order, per method. */
function keptCategories(md: PruningMetadata): Category[] {
  if (md.method === "random") {
    return [
      {
        name: "random",
        indices: md.uniform ?? [],
        color: UNIFORM,
        ranked: false,
      },
    ];
  }
  if (md.method === "nprune" || md.method === "checkered" || md.uniform) {
    // Lattice / checkerboard: a single unranked keep set with no
    // scores or pick order.
    return [
      {
        name: "uniform",
        indices: md.uniform ?? [],
        color: UNIFORM,
        ranked: false,
      },
    ];
  }
  if (md.method === "dart" || md.pivots) {
    return [
      { name: "pivots", indices: md.pivots ?? [], color: PIVOT, ranked: true },
      { name: "diverse", indices: md.diverse ?? [], color: DIVERSE, ranked: true },
    ];
  }
  // AnchorPrune: `expansion` is its distinguishing field (plain HiPrune
  // also reports `anchors`).
  if (md.method === "anchorprune" || md.expansion) {
    return [
      {
        name: "anchors",
        indices: md.anchors ?? [],
        color: ANCHOR,
        ranked: true,
      },
      {
        name: "expansion",
        indices: md.expansion ?? [],
        color: DIVERSE,
        ranked: true,
      },
    ];
  }
  const cats: Category[] = [
    { name: "anchors", indices: md.anchors ?? [], color: ANCHOR, ranked: true },
    { name: "buffers", indices: md.buffers ?? [], color: BUFFER, ranked: false },
  ];
  if (md.method === "hydart" || md.diverse) {
    cats.push({
      name: "diverse",
      indices: md.diverse ?? [],
      color: DIVERSE,
      ranked: true,
    });
    return cats;
  }
  cats.push({
    name: "registers",
    indices: md.registers ?? [],
    color: REGISTER,
    ranked: true,
  });
  if (md.method === "hiprune_pp" || md.prompt_tokens) {
    cats.push({
      name: "prompt",
      indices: md.prompt_tokens ?? [],
      color: PROMPT,
      ranked: true,
    });
  }
  return cats;
}

/** Brief blurbs for legend chips — matches the overlay hover tone. */
const CATEGORY_BLURB: Record<string, string> = {
  anchors:
    "Patches with the highest average attention in the middle layer of the vision encoder.",
  buffers: "The neighbors of the anchor patch(es).",
  registers:
    "Remaining patches with the highest average attention received in the final layer of the vision encoder.",
  prompt:
    "Patches in the final layer with the highest cosine similarity to the averaged text embedding.",
  pruned: "Dropped visual patches — not sent to the language model.",
  diverse:
    "Budget fillers chosen to stay dissimilar from tokens already kept.",
  pivots:
    "Early LLM-attention pivots that seed the duplication test (DART).",
  expansion:
    "Important-but-novel patches that expand around prompt-relevant anchors.",
  random:
    "A random subset of visual patches kept at the retention ratio. Prompt-agnostic — click Random again to draw a new subset.",
  uniform: "A fixed lattice or checkerboard keep pattern (no scores or ranking).",
};

/** Singular tooltip label for a legend/category name. */
function singular(name: string): string {
  return name === "pivots"
    ? "pivot"
    : name === "anchors"
      ? "anchor"
      : name === "buffers"
        ? "buffer"
        : name === "registers"
          ? "register"
          : name;
}

/** Draw the uploaded image the way the model's preprocessor sees it:
 * LLaVA (llava-hf) resizes the shortest edge to 336 then center-crops a
 * square; Qwen resizes to the grid's aspect (its 28px-multiple resize is
 * close enough to a plain stretch for visualization). */
function drawBase(
  ctx: CanvasRenderingContext2D,
  img: HTMLImageElement,
  w: number,
  h: number,
  model: ModelKey
) {
  if (model === "llava_1_5") {
    const side = Math.min(img.width, img.height);
    const sx = (img.width - side) / 2;
    const sy = (img.height - side) / 2;
    ctx.drawImage(img, sx, sy, side, side, 0, 0, w, h);
  } else {
    ctx.drawImage(img, 0, 0, w, h);
  }
}

interface CellInfo {
  category: string;
  color: string;
  /** 1-based rank within the category, for ordered categories
   * (anchors/registers: attention top-k order; diverse: pick order). */
  rank: number | null;
  rankOf: number | null;
}

/** index -> category/rank lookup for the tooltip. */
function buildCellIndex(md: PruningMetadata): Map<number, CellInfo> {
  const map = new Map<number, CellInfo>();
  const ordered: Array<[string, number[], string, boolean]> = [
    ...keptCategories(md).map(
      (c): [string, number[], string, boolean] => [
        singular(c.name),
        c.indices,
        c.color,
        c.ranked,
      ]
    ),
    ["pruned", md.pruned, "rgb(120,113,108)", false],
  ];
  for (const [category, indices, color, ranked] of ordered) {
    indices.forEach((idx, i) => {
      map.set(idx, {
        category,
        color,
        rank: ranked ? i + 1 : null,
        rankOf: ranked ? indices.length : null,
      });
    });
  }
  return map;
}

function fmtScore(v: number | undefined, uniform: number): string {
  if (v == null) return "—";
  return `${v.toExponential(2)} (${(v / uniform).toFixed(2)}x uniform)`;
}

export function OverlayCanvas({
  imageUrl,
  metadata,
  model,
  showHeatmap = false,
  heatLayerIdx = 0,
  heatStyle = "mask",
  /** When set with a color heatmap, only paint this many cells (attn-ranked). */
  heatRevealCount,
  /** Duration for kept-token outline reveal (pruned overlay). */
  keptRevealMs = 700,
  /** When true, dark pruned patches also lay in progressively (demo). */
  animatePrunedPatches = false,
  fill = false,
}: {
  imageUrl: string;
  metadata: PruningMetadata;
  model: ModelKey;
  showHeatmap?: boolean;
  /** 0-based vision-encoder layer when scrubbing vision_layers. */
  heatLayerIdx?: number;
  /** `mask` = dark reveal (default UI). `color` = opaque jet colormap (demo). */
  heatStyle?: "mask" | "color";
  heatRevealCount?: number;
  keptRevealMs?: number;
  animatePrunedPatches?: boolean;
  /** Absolute-fill a sized parent (CompareFrame). */
  fill?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [hover, setHover] = useState<{
    idx: number;
    xPct: number;
    yPct: number;
  } | null>(null);
  const cellIndex = useMemo(() => buildCellIndex(metadata), [metadata]);
  const visionLayers = metadata.scores?.vision_layers;
  const layerScores =
    visionLayers &&
    visionLayers.length > 0 &&
    heatLayerIdx >= 0 &&
    heatLayerIdx < visionLayers.length
      ? visionLayers[heatLayerIdx]
      : undefined;
  const heatScores = layerScores ?? metadata.scores?.object_layer;
  const canHeatmap = Boolean(
    heatScores && heatScores.length === metadata.num_tokens
  );
  const heatmapOn = showHeatmap && canHeatmap;

  /** High-attention cells first — matches “patches laid in sequence.” */
  const heatOrder = useMemo(() => {
    if (!heatScores) return [] as number[];
    return heatScores
      .map((score, idx) => ({ score, idx }))
      .sort((a, b) => b.score - a.score)
      .map((row) => row.idx);
  }, [heatScores]);

  // Kept outlines in category draw order — revealed one (or a few) at a
  // time so retention feels like the answer typewriter, not a hard cut.
  const keptStrokes = useMemo(() => {
    const out: Array<{ idx: number; color: string }> = [];
    for (const c of keptCategories(metadata)) {
      for (const idx of c.indices) out.push({ idx, color: c.color });
    }
    return out;
  }, [metadata]);

  const prunedIndices = metadata.pruned;
  const [revealedKept, setRevealedKept] = useState(keptStrokes.length);
  const [revealedPruned, setRevealedPruned] = useState(prunedIndices.length);

  useEffect(() => {
    const prefersReduced =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const nKept = keptStrokes.length;
    const nPruned = prunedIndices.length;
    if (heatmapOn || prefersReduced) {
      setRevealedKept(nKept);
      setRevealedPruned(nPruned);
      return;
    }
    setRevealedPruned(animatePrunedPatches ? 0 : nPruned);
    setRevealedKept(0);
    let iKept = 0;
    let iPruned = 0;
    const duration = Math.max(200, keptRevealMs);
    const ticks = 36;
    const keptStep = Math.max(1, Math.ceil(Math.max(nKept, 1) / ticks));
    const prunedStep = Math.max(1, Math.ceil(Math.max(nPruned, 1) / ticks));
    const steps = Math.max(
      Math.ceil(nKept / keptStep),
      animatePrunedPatches ? Math.ceil(nPruned / prunedStep) : 0,
      1
    );
    const intervalMs = Math.min(48, Math.max(16, duration / steps));
    const id = window.setInterval(() => {
      let done = true;
      if (animatePrunedPatches && iPruned < nPruned) {
        iPruned = Math.min(nPruned, iPruned + prunedStep);
        setRevealedPruned(iPruned);
        done = false;
      }
      if (iKept < nKept) {
        iKept = Math.min(nKept, iKept + keptStep);
        setRevealedKept(iKept);
        done = false;
      }
      if (done) window.clearInterval(id);
    }, intervalMs);
    return () => window.clearInterval(id);
  }, [keptStrokes, prunedIndices, heatmapOn, keptRevealMs, animatePrunedPatches]);

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !img.complete) return;
    const [gridW, gridH] = metadata.grid;
    const cell = Math.max(6, Math.round(700 / Math.max(gridW, gridH)));
    const nextW = gridW * cell;
    const nextH = gridH * cell;
    // Assigning width/height clears the bitmap — only do it when size changes
    // so layer scrubbing does not flash a blank frame.
    if (canvas.width !== nextW || canvas.height !== nextH) {
      canvas.width = nextW;
      canvas.height = nextH;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    drawBase(ctx, img, canvas.width, canvas.height, model);

    const box = (idx: number): [number, number] => [
      (idx % gridW) * cell,
      Math.floor(idx / gridW) * cell,
    ];

    if (heatmapOn && heatScores) {
      const norm = normalizeScores(heatScores);
      const total = heatOrder.length || norm.length;
      const count =
        heatRevealCount == null
          ? total
          : Math.max(0, Math.min(total, heatRevealCount));
      // Progressive color mode: lay patches on the photo (no full-frame dim).
      // Mask mode / full reveal: keep the existing dark-reveal look.
      if (heatStyle === "mask" || heatRevealCount == null) {
        if (heatStyle === "color") {
          ctx.fillStyle = "rgba(12,10,9,0.28)";
          ctx.fillRect(0, 0, canvas.width, canvas.height);
        }
      }
      // Demo color heatmap: continuous jet for most cells, then paint
      // every revealed anchor solid hot-red on top so the top-k hottest
      // patches read as a matched set (not just the single peak).
      const hotAnchors = metadata.anchors ?? [];
      const revealed =
        heatRevealCount == null
          ? null
          : new Set(heatOrder.slice(0, count));
      for (let i = 0; i < count; i++) {
        const idx = heatOrder[i] ?? i;
        const [x, y] = box(idx);
        if (heatStyle === "color") {
          ctx.fillStyle = heatColorFill(norm[idx]);
          ctx.fillRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
        } else {
          ctx.fillStyle = heatFill(norm[idx]);
          ctx.fillRect(x, y, cell, cell);
        }
      }
      if (heatStyle === "color" && hotAnchors.length > 0) {
        ctx.fillStyle = heatColorFill(1);
        for (const idx of hotAnchors) {
          if (revealed && !revealed.has(idx)) continue;
          const [x, y] = box(idx);
          ctx.fillRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
        }
      }
    } else {
      const prunedCount = Math.max(
        0,
        Math.min(prunedIndices.length, revealedPruned)
      );
      ctx.fillStyle = PRUNED_FILL;
      for (let i = 0; i < prunedCount; i++) {
        const idx = prunedIndices[i];
        const [x, y] = box(idx);
        ctx.fillRect(x, y, cell, cell);
      }
      // Light grid on pruned cells so adjacent blacks still read as
      // individual soft-token squares instead of one solid blob.
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(255,255,255,0.18)";
      for (let i = 0; i < prunedCount; i++) {
        const idx = prunedIndices[i];
        const [x, y] = box(idx);
        ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
      }

      ctx.lineWidth = 2;
      for (let i = 0; i < revealedKept; i++) {
        const stroke = keptStrokes[i];
        if (!stroke) break;
        const [x, y] = box(stroke.idx);
        ctx.strokeStyle = stroke.color;
        ctx.strokeRect(x + 1, y + 1, cell - 2, cell - 2);
      }
    }
  }, [
    metadata,
    model,
    heatmapOn,
    heatScores,
    heatStyle,
    heatOrder,
    heatRevealCount,
    keptStrokes,
    revealedKept,
    prunedIndices,
    revealedPruned,
  ]);

  // Load the source image once per URL; keep it for fast layer repaints.
  useEffect(() => {
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      imgRef.current = img;
      paint();
    };
    img.onerror = () => {
      if (!cancelled) imgRef.current = null;
    };
    img.src = imageUrl;
    return () => {
      cancelled = true;
    };
  }, [imageUrl, paint]);

  // Repaint synchronously when overlay inputs change (no image reload).
  useEffect(() => {
    paint();
  }, [paint]);

  const [gridW, gridH] = metadata.grid;
  const uniform = 1 / metadata.num_tokens;

  const onMove = (e: ReactMouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    const col = Math.min(gridW - 1, Math.max(0, Math.floor(x * gridW)));
    const row = Math.min(gridH - 1, Math.max(0, Math.floor(y * gridH)));
    setHover({ idx: row * gridW + col, xPct: x * 100, yPct: y * 100 });
  };

  const info = hover ? cellIndex.get(hover.idx) : undefined;
  const layerAttn = hover && heatScores ? heatScores[hover.idx] : undefined;
  const objScore =
    hover && !layerScores
      ? metadata.scores?.object_layer?.[hover.idx]
      : undefined;
  const deepScore = hover ? metadata.scores?.deep_layer?.[hover.idx] : undefined;
  const simScore = hover ? metadata.scores?.similarity?.[hover.idx] : undefined;
  const textSimScore = hover
    ? metadata.scores?.text_similarity?.[hover.idx]
    : undefined;
  const keyNorm = hover ? metadata.scores?.key_norm?.[hover.idx] : undefined;
  const pivotSim = hover
    ? metadata.scores?.pivot_similarity?.[hover.idx]
    : undefined;
  const hoverRow = hover ? Math.floor(hover.idx / gridW) : 0;
  const hoverCol = hover ? hover.idx % gridW : 0;

  return (
    <div className={fill ? "absolute inset-0" : "relative"}>
      <canvas
        ref={canvasRef}
        className={
          fill
            ? "h-full w-full block"
            : "w-full h-auto block border border-border"
        }
        style={{
          borderRadius: fill ? undefined : "var(--r-1)",
          cursor: "crosshair",
        }}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      />
      {hover && (
        <div
          aria-hidden
          className="absolute pointer-events-none border-2 border-white mix-blend-difference"
          style={{
            left: `${(hoverCol / gridW) * 100}%`,
            top: `${(hoverRow / gridH) * 100}%`,
            width: `${100 / gridW}%`,
            height: `${100 / gridH}%`,
          }}
        />
      )}
      {hover && (info || heatmapOn) && (
        <div
          className="absolute z-10 pointer-events-none bg-stone-950 text-white p-2.5 flex flex-col gap-1"
          style={{
            borderRadius: "var(--r-1)",
            left: hover.xPct < 55 ? `calc(${hover.xPct}% + 14px)` : undefined,
            right: hover.xPct >= 55 ? `calc(${100 - hover.xPct}% + 14px)` : undefined,
            top: `min(${hover.yPct}%, calc(100% - 96px))`,
            maxWidth: 260,
          }}
        >
          <span className="demo-label" style={{ color: "#a8a29e" }}>
            token {hover.idx} — row {hoverRow}, col {hoverCol}
          </span>
          {!heatmapOn && info && (
            <span className="text-xs font-mono flex items-center gap-1.5">
              <span
                aria-hidden
                className="inline-block shrink-0"
                style={{
                  width: 8,
                  height: 8,
                  background: info.color,
                  borderRadius: 1,
                }}
              />
              {info.category}
              {info.rank != null && ` — rank ${info.rank}/${info.rankOf}`}
            </span>
          )}
          {layerAttn != null && layerScores && (
            <span className="text-xs font-mono" style={{ color: "#d6d3d1" }}>
              layer {heatLayerIdx + 1} attn: {fmtScore(layerAttn, uniform)}
            </span>
          )}
          {objScore != null && (
            <span className="text-xs font-mono" style={{ color: "#d6d3d1" }}>
              obj attn: {fmtScore(objScore, uniform)}
            </span>
          )}
          {keyNorm != null && (
            <span className="text-xs font-mono" style={{ color: "#d6d3d1" }}>
              key L1 norm: {keyNorm.toFixed(1)}
            </span>
          )}
          {pivotSim != null && (
            <span className="text-xs font-mono" style={{ color: "#d6d3d1" }}>
              pivot cos sim: {pivotSim.toFixed(3)}
            </span>
          )}
          {deepScore != null && (
            <span className="text-xs font-mono" style={{ color: "#d6d3d1" }}>
              deep attn: {fmtScore(deepScore, uniform)}
            </span>
          )}
          {simScore != null && (
            <span className="text-xs font-mono" style={{ color: "#d6d3d1" }}>
              max cos sim: {simScore.toFixed(3)}
            </span>
          )}
          {textSimScore != null && (
            <span className="text-xs font-mono" style={{ color: "#d6d3d1" }}>
              text cos sim: {textSimScore.toFixed(3)}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

/** Kept-token color chips plus a black pruned chip. Heatmap toggle +
 *  layer scrub live in the side panel. */
function LegendChip({
  name,
  color,
  count,
}: {
  name: string;
  color: string;
  count: number;
}) {
  const blurb = CATEGORY_BLURB[name];
  return (
    <span className="relative inline-flex items-center gap-1.5 demo-label text-fg-muted group/leg cursor-help">
      <span
        aria-hidden
        className="inline-block"
        style={{ width: 8, height: 8, background: color, borderRadius: 1 }}
      />
      {name} {count}
      {blurb && (
        <span
          role="tooltip"
          className={
            "pointer-events-none absolute left-0 bottom-full mb-2 z-20 " +
            "hidden group-hover/leg:flex flex-col gap-1 " +
            "bg-stone-950 text-white p-2.5 w-56 max-w-[min(14rem,70vw)]"
          }
          style={{ borderRadius: "var(--r-1)" }}
        >
          <span className="text-xs font-mono flex items-center gap-1.5">
            <span
              aria-hidden
              className="inline-block shrink-0"
              style={{
                width: 8,
                height: 8,
                background: color,
                borderRadius: 1,
              }}
            />
            {singular(name)}
          </span>
          <span className="text-xs leading-snug" style={{ color: "#d6d3d1" }}>
            {blurb}
          </span>
        </span>
      )}
    </span>
  );
}

export function OverlayLegend({
  metadata,
  showHeatmap,
}: {
  metadata: PruningMetadata;
  showHeatmap: boolean;
}) {
  const canHeatmap = Boolean(
    metadata.scores?.object_layer &&
      metadata.scores.object_layer.length === metadata.num_tokens
  );

  if (showHeatmap && canHeatmap) {
    return (
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        <span className="inline-flex items-center gap-1.5 demo-label text-fg-muted">
          <span
            aria-hidden
            className="inline-block"
            style={{
              width: 48,
              height: 8,
              borderRadius: 1,
              background:
                "linear-gradient(90deg, rgba(12,10,9,0.82), rgba(12,10,9,0))",
            }}
          />
          low attention → high attention
        </span>
      </div>
    );
  }

  const items = keptCategories(metadata)
    .filter((c) => c.indices.length > 0)
    .map(
      (c): [string, string, number] => [c.name, c.color, c.indices.length]
    );
  // HiPrune++ already has four kept categories (anchors/buffers/registers/
  // prompt); the pruned chip makes the legend wrap under the image.
  if (metadata.pruned.length > 0 && metadata.method !== "hiprune_pp") {
    items.push(["pruned", "rgb(12,10,9)", metadata.pruned.length]);
  }
  if (items.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
      {items.map(([name, color, count]) => (
        <LegendChip key={name} name={name} color={color} count={count} />
      ))}
    </div>
  );
}

/** True when the last result can show an attention heatmap overlay. */
export function canShowHeatmap(
  metadata: PruningMetadata | null | undefined
): boolean {
  if (!metadata?.scores?.object_layer) return false;
  return metadata.scores.object_layer.length === metadata.num_tokens;
}

/** Number of scrubbable vision-encoder layers, or 0 if scrubbing is unavailable. */
export function heatLayerCount(
  metadata: PruningMetadata | null | undefined
): number {
  const layers = metadata?.scores?.vision_layers;
  if (!metadata || !layers?.length) return 0;
  if (!layers.every((row) => row.length === metadata.num_tokens)) return 0;
  return layers.length;
}

export function defaultHeatLayerIdx(
  metadata: PruningMetadata | null | undefined
): number {
  if (!metadata?.scores) return 0;
  const idx = metadata.scores.vision_layer_object_idx;
  const layers = metadata.scores.vision_layers;
  if (
    typeof idx === "number" &&
    layers &&
    idx >= 0 &&
    idx < layers.length
  ) {
    return idx;
  }
  return 0;
}

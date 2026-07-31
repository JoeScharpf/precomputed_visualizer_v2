/**
 * Loop:
 * 1) Original alone, centered
 * 2) Slides left + attention heatmap starts together
 * 3) Pruned fades in on the right with pruned/kept overlay
 * 4) Stay side-by-side: left pulses original ↔ attention; right stays pruned
 *
 * Keys: number once selects a pack (static), same number again starts
 * the animation. R restarts, Esc idle.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { OverlayCanvas } from "./ImageCompare";
import type { InferResult, ModelKey, PruningMetadata } from "./lib/types";

const CENTER_HOLD_MS = 1800;
/** Slide duration — keep in sync with `.loop-move` in index.css */
const SLIDE_MS = 800;
const PRUNED_FADE_MS = 1100;
const HEAT_REVEAL_MS = 1600;
const PRUNED_REVEAL_MS = 1400;
/** Brief hold at full heatmap / plain original while pulsing. */
const HEAT_PULSE_HOLD_MS = 1800;
const PANEL_MAX_H = 480;
const PANEL_GAP = 32;

const PACK_KEYS: Record<string, number> = {
  "1": 0,
  "2": 1,
  "3": 2,
  "4": 3,
  "5": 4,
  "6": 5,
  "7": 6,
};

type PackManifestEntry = {
  id: string;
  label: string;
  image: string;
  pack: string;
};

type DemoPack = {
  id: string;
  label: string;
  prompt: string;
  method: string;
  retention: number;
  model: ModelKey;
  image: string;
  result: InferResult;
};

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const id = window.setTimeout(resolve, ms);
    const onAbort = () => {
      window.clearTimeout(id);
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function nextFrame(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => {
      requestAnimationFrame(() => resolve());
    });
  });
}

function PanelFrame({
  width,
  height,
  children,
}: {
  width: number;
  height: number;
  children: ReactNode;
}) {
  return (
    <div
      className="relative overflow-hidden bg-stone-950"
      style={{ width, height }}
    >
      {children}
    </div>
  );
}

function PlainImage({ src }: { src: string }) {
  return (
    <img
      src={src}
      alt=""
      className="h-full w-full object-fill"
      style={{ display: "block" }}
      draggable={false}
    />
  );
}

async function ensureImageLoaded(src: string, signal?: AbortSignal) {
  if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
  const img = new Image();
  img.decoding = "async";
  img.src = src;
  try {
    if (typeof img.decode === "function") {
      await img.decode();
    } else {
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error(`Failed to load ${src}`));
      });
    }
  } catch {
    /* ignore */
  }
  if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
}

export default function DemoApp() {
  const [manifest, setManifest] = useState<PackManifestEntry[]>([]);
  const [packCache, setPackCache] = useState<Record<string, DemoPack>>({});
  const [loadError, setLoadError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  const [active, setActive] = useState<DemoPack | null>(null);
  /** False = pack armed/static (recording cue); true = animation running. */
  const [playing, setPlaying] = useState(false);
  /** Original at center (true) vs left column (false). */
  const [atCenter, setAtCenter] = useState(true);
  /** CSS transition only while sliding center → left. */
  const [slideOn, setSlideOn] = useState(false);
  /** Right panel mounted (only after slide finishes). */
  const [showRight, setShowRight] = useState(false);
  const [prunedIn, setPrunedIn] = useState(false);
  const [heatRevealCount, setHeatRevealCount] = useState<number | null>(null);
  const [showPrunedOverlay, setShowPrunedOverlay] = useState(false);
  const [loopEpoch, setLoopEpoch] = useState(0);
  const [playEpoch, setPlayEpoch] = useState(0);
  const [viewportW, setViewportW] = useState(() =>
    typeof window !== "undefined" ? window.innerWidth : 1200
  );
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const onResize = () => setViewportW(window.innerWidth);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const res = await fetch("/demo-packs/manifest.json");
        if (!res.ok) throw new Error(`Could not load demo packs (${res.status})`);
        const data = (await res.json()) as { packs: PackManifestEntry[] };
        const entries = data.packs ?? [];
        if (!alive) return;
        setManifest(entries);

        const loaded: Record<string, DemoPack> = {};
        await Promise.all(
          entries.map(async (entry) => {
            const packRes = await fetch(entry.pack);
            if (!packRes.ok) {
              throw new Error(`Could not load ${entry.label} pack`);
            }
            const pack = (await packRes.json()) as DemoPack;
            if (!pack.result?.metadata) {
              throw new Error(`Pack ${entry.label} is missing metadata`);
            }
            loaded[entry.id] = pack;
            void ensureImageLoaded(pack.image);
          })
        );
        if (!alive) return;
        setPackCache(loaded);
        setReady(true);
        // Arm first pack static — press its number again to start.
        const first = entries[0] && loaded[entries[0].id];
        if (first) {
          setActive(first);
          setPlaying(false);
        }
      } catch (e) {
        if (alive) {
          setLoadError(e instanceof Error ? e.message : String(e));
        }
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const hardReset = useCallback(() => {
    setSlideOn(false);
    setAtCenter(true);
    setShowRight(false);
    setPrunedIn(false);
    setHeatRevealCount(null);
    setShowPrunedOverlay(false);
  }, []);

  const stopPlayback = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    hardReset();
    setPlaying(false);
    setActive(null);
  }, [hardReset]);

  /** Select a pack and freeze on the centered original (no animation). */
  const selectPackAtIndex = useCallback(
    (index: number) => {
      const entry = manifest[index];
      if (!entry) return;
      const pack = packCache[entry.id];
      if (!pack) return;
      abortRef.current?.abort();
      abortRef.current = null;
      setLoadError(null);
      hardReset();
      setPlaying(false);
      setActive(pack);
    },
    [manifest, packCache, hardReset]
  );

  const startPlayback = useCallback(() => {
    if (!active) return;
    setPlaying(true);
    setPlayEpoch((n) => n + 1);
  }, [active]);

  /** Stop animation but keep the pack armed on the static original. */
  const armCurrentPack = useCallback(() => {
    if (!active) return;
    abortRef.current?.abort();
    abortRef.current = null;
    hardReset();
    setPlaying(false);
  }, [active, hardReset]);

  const animateHeatReveal = async (
    from: number,
    to: number,
    durationMs: number,
    signal: AbortSignal
  ) => {
    if (durationMs <= 0) {
      setHeatRevealCount(to);
      return;
    }
    const ticks = 36;
    const intervalMs = durationMs / ticks;
    for (let t = 1; t <= ticks; t++) {
      if (signal.aborted) {
        throw new DOMException("Aborted", "AbortError");
      }
      setHeatRevealCount(Math.round(from + (to - from) * (t / ticks)));
      await sleep(intervalMs, signal);
    }
    setHeatRevealCount(to);
  };

  useEffect(() => {
    if (!active || !playing) return;

    const md = active.result.metadata;
    const heatN =
      md?.scores?.object_layer &&
      md.scores.object_layer.length === md.num_tokens
        ? md.num_tokens
        : 0;

    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    const { signal } = ac;

    const play = async () => {
      try {
        await ensureImageLoaded(active.image, signal);

        // Intro once: center → slide+heat → pruned pair
        hardReset();
        setLoopEpoch((n) => n + 1);
        await nextFrame();
        if (signal.aborted) return;

        await sleep(CENTER_HOLD_MS, signal);

        setSlideOn(true);
        await nextFrame();
        setAtCenter(false);
        if (heatN > 0) {
          setHeatRevealCount(0);
          await nextFrame();
          await Promise.all([
            sleep(SLIDE_MS, signal),
            animateHeatReveal(0, heatN, HEAT_REVEAL_MS, signal),
          ]);
        } else {
          await sleep(SLIDE_MS, signal);
        }

        setShowRight(true);
        setShowPrunedOverlay(true);
        setPrunedIn(false);
        await nextFrame();
        setPrunedIn(true);
        await sleep(Math.max(PRUNED_FADE_MS, PRUNED_REVEAL_MS), signal);
        await sleep(HEAT_PULSE_HOLD_MS, signal);

        // Stay paired: left pulses attention on/off; right stays put
        if (heatN <= 0) return;

        while (!signal.aborted) {
          // Attention → original
          await animateHeatReveal(heatN, 0, HEAT_REVEAL_MS, signal);
          setHeatRevealCount(null);
          await sleep(HEAT_PULSE_HOLD_MS, signal);

          // Original → attention
          setHeatRevealCount(0);
          await nextFrame();
          await animateHeatReveal(0, heatN, HEAT_REVEAL_MS, signal);
          await sleep(HEAT_PULSE_HOLD_MS, signal);
        }
      } catch (e) {
        if (e instanceof DOMException && e.name === "AbortError") return;
        throw e;
      }
    };

    void play();
    return () => {
      ac.abort();
    };
  }, [active, playEpoch, playing, hardReset]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      const packIdx = PACK_KEYS[e.key];
      if (packIdx != null) {
        e.preventDefault();
        if (!ready) return;
        const entry = manifest[packIdx];
        if (!entry) return;
        // Same pack: start if armed, re-arm (static) if already playing.
        if (active?.id === entry.id) {
          if (playing) armCurrentPack();
          else startPlayback();
          return;
        }
        selectPackAtIndex(packIdx);
        return;
      }
      if (e.key.toLowerCase() === "r") {
        if (!active) return;
        e.preventDefault();
        startPlayback();
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        stopPlayback();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [
    ready,
    active,
    playing,
    manifest,
    selectPackAtIndex,
    startPlayback,
    armCurrentPack,
    stopPlayback,
  ]);

  const imageUrl = active?.image ?? null;
  const md = (active?.result?.metadata ?? null) as PruningMetadata | null;
  const model: ModelKey = active?.model ?? "gemma4";
  const gridW = md?.grid?.[0];
  const gridH = md?.grid?.[1];
  const canHeatmap = Boolean(
    md?.scores?.object_layer &&
      md.scores.object_layer.length === md.num_tokens
  );

  const layout = useMemo(() => {
    if (!gridW || !gridH) return null;
    const panelH = PANEL_MAX_H;
    const panelW = Math.round(PANEL_MAX_H * (gridW / gridH));
    const stageW = panelW * 2 + PANEL_GAP;
    const centerShift = (panelW + PANEL_GAP) / 2;
    const scale = Math.min(1, (viewportW - 32) / stageW);
    return { panelW, panelH, stageW, centerShift, scale };
  }, [gridW, gridH, viewportW]);

  return (
    <div
      className="h-screen overflow-hidden flex flex-col"
      style={{ background: "#FAFAF9" }}
    >
      <main className="flex-1 min-h-0 flex items-center justify-center px-4 overflow-hidden">
        {loadError && !active && (
          <p className="text-sm border border-red-200 bg-red-50 text-red-700 p-2 max-w-md">
            {loadError}
          </p>
        )}

        {active && imageUrl && md && layout && (
          <div
            style={{
              width: layout.stageW * layout.scale,
              height: layout.panelH * layout.scale,
            }}
          >
            <div
              className="relative"
              style={{
                width: layout.stageW,
                height: layout.panelH,
                transform: `scale(${layout.scale})`,
                transformOrigin: "top left",
              }}
            >
              {/* Single original panel — translateX center ↔ left */}
              <div
                key={`orig-${loopEpoch}`}
                className={slideOn ? "loop-move" : "loop-move--snap"}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: layout.panelW,
                  height: layout.panelH,
                  transform: atCenter
                    ? `translateX(${layout.centerShift}px)`
                    : "translateX(0px)",
                }}
              >
                <PanelFrame width={layout.panelW} height={layout.panelH}>
                  <PlainImage src={imageUrl} />
                  {canHeatmap && heatRevealCount != null && (
                    <div className="absolute inset-0" aria-hidden>
                      <OverlayCanvas
                        imageUrl={imageUrl}
                        metadata={md}
                        model={model}
                        showHeatmap
                        heatStyle="color"
                        heatRevealCount={heatRevealCount}
                        heatLayerIdx={0}
                        fill
                      />
                    </div>
                  )}
                </PanelFrame>
              </div>

              {/* Right panel only after slide — never overlaps the solo phase */}
              {showRight && (
                <div
                  className={
                    "absolute top-0 " +
                    (prunedIn ? "loop-fade loop-fade--in" : "loop-fade")
                  }
                  style={{
                    width: layout.panelW,
                    height: layout.panelH,
                    left: layout.panelW + PANEL_GAP,
                  }}
                >
                  <PanelFrame width={layout.panelW} height={layout.panelH}>
                    <PlainImage src={imageUrl} />
                    {showPrunedOverlay && (
                      <div className="absolute inset-0" aria-hidden>
                        <OverlayCanvas
                          key={`pruned-${loopEpoch}`}
                          imageUrl={imageUrl}
                          metadata={md}
                          model={model}
                          showHeatmap={false}
                          heatLayerIdx={0}
                          keptRevealMs={PRUNED_REVEAL_MS}
                          animatePrunedPatches
                          fill
                        />
                      </div>
                    )}
                  </PanelFrame>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

import { useState } from "react";

import type { IdentifyResponse } from "../lib/types";

type ResultsPanelProps = {
  result: IdentifyResponse | null;
  isLoading: boolean;
  error: string | null;
};

function ConfidenceBar({ confidence }: { confidence: number }) {
  const percent = Math.max(0, Math.min(100, Math.round(confidence * 100)));
  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-xs uppercase tracking-[0.24em] text-slate-400">
        <span>Confidence</span>
        <span>{percent}%</span>
      </div>
      <div className="h-2 rounded-full bg-white/10">
        <div
          className="h-2 rounded-full bg-gradient-to-r from-aurora to-gold"
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}

export function ResultsPanel({ result, isLoading, error }: ResultsPanelProps) {
  const [zoomPercent, setZoomPercent] = useState(100);

  return (
    <section className="rounded-[2rem] border border-white/10 bg-white/5 p-5 shadow-panel backdrop-blur md:p-7">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm uppercase tracking-[0.28em] text-slate-400">Analysis</p>
          <h2 className="mt-2 font-display text-3xl text-white">Detection Results</h2>
        </div>
        {isLoading ? (
          <div className="h-10 w-10 animate-spin rounded-full border-2 border-white/20 border-t-aurora" />
        ) : null}
      </div>

      {error ? (
        <div className="mt-5 rounded-2xl border border-red-400/30 bg-red-500/10 p-4 text-sm text-red-100">
          {error}
        </div>
      ) : null}

      {!result && !isLoading && !error ? (
        <div className="mt-6 rounded-[1.5rem] border border-white/10 bg-night-900/50 p-5 text-sm text-slate-300">
          Upload a night-sky photo to detect star fields, probable planets, and supported constellations.
        </div>
      ) : null}

      {result ? (
        <div className="mt-6 grid gap-5 xl:grid-cols-[1.45fr_0.95fr]">
          <div className="rounded-[1.5rem] border border-white/10 bg-night-900/80">
            <div className="flex items-center justify-between gap-4 border-b border-white/10 px-4 py-3">
              <p className="text-sm uppercase tracking-[0.28em] text-slate-400">Annotated Image</p>
              <div className="flex items-center gap-3">
                <label htmlFor="zoom" className="text-xs uppercase tracking-[0.2em] text-slate-400">
                  Zoom
                </label>
                <input
                  id="zoom"
                  type="range"
                  min={50}
                  max={200}
                  step={10}
                  value={zoomPercent}
                  onChange={(event) => setZoomPercent(Number(event.target.value))}
                  className="w-28 accent-aurora"
                />
                <span className="w-12 text-right text-sm text-slate-300">{zoomPercent}%</span>
              </div>
            </div>

            <div className="max-h-[720px] overflow-auto p-4">
              <img
                src={`data:image/png;base64,${result.annotated_image}`}
                alt="Annotated constellation analysis"
                className="h-auto max-w-none rounded-xl"
                style={{ width: `${zoomPercent}%` }}
              />
            </div>
          </div>

          <div className="space-y-4">
            <div className="rounded-[1.5rem] border border-white/10 bg-night-900/70 p-5">
              <p className="text-sm uppercase tracking-[0.28em] text-slate-400">Summary</p>
              <div className="mt-4 grid grid-cols-2 gap-4">
                <div className="rounded-2xl bg-white/5 p-4">
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Stars</p>
                  <p className="mt-2 text-3xl font-semibold text-white">{result.stars_detected}</p>
                </div>
                <div className="rounded-2xl bg-white/5 p-4">
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Planets</p>
                  <p className="mt-2 text-3xl font-semibold text-white">
                    {result.possible_planets.length}
                  </p>
                </div>
              </div>
            </div>

            <div className="rounded-[1.5rem] border border-white/10 bg-night-900/70 p-5">
              <p className="text-sm uppercase tracking-[0.28em] text-slate-400">Constellations</p>
              <div className="mt-4 space-y-4">
                {result.constellations.length > 0 ? (
                  result.constellations.map((item) => (
                    <div key={item.name} className="rounded-2xl bg-white/5 p-4">
                      <div className="mb-3 flex items-center justify-between gap-4">
                        <p className="font-semibold text-white">{item.name}</p>
                        <span className="text-sm text-slate-300">
                          {(item.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <ConfidenceBar confidence={item.confidence} />
                    </div>
                  ))
                ) : (
                  <p className="text-sm text-slate-300">
                    No supported constellation passed the confidence threshold for this image.
                  </p>
                )}
              </div>
            </div>

            <div className="rounded-[1.5rem] border border-white/10 bg-night-900/70 p-5">
              <p className="text-sm uppercase tracking-[0.28em] text-slate-400">Possible Bright Objects</p>
              <div className="mt-4 space-y-3 text-sm text-slate-300">
                {result.possible_planets.length > 0 ? (
                  result.possible_planets.slice(0, 5).map((planet, index) => (
                    <div
                      key={`${planet.x}-${planet.y}-${index}`}
                      className="flex items-center justify-between rounded-2xl bg-white/5 px-4 py-3"
                    >
                      <span>
                        X {planet.x.toFixed(0)}, Y {planet.y.toFixed(0)}
                      </span>
                      <span>Brightness {planet.brightness.toFixed(1)}</span>
                    </div>
                  ))
                ) : (
                  <p>No unusually bright objects were flagged.</p>
                )}
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}

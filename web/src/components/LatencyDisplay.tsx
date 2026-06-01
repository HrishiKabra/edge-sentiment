/**
 * Instrument-style latency readout: shows the active backend and the last
 * measured inference latency, with a pulsing "live" dot while inference runs.
 */
interface LatencyDisplayProps {
  backendLabel: string;
  latencyMs: number | null;
  running: boolean;
}

export function LatencyDisplay({ backendLabel, latencyMs, running }: LatencyDisplayProps) {
  return (
    <div className="latency" data-running={running}>
      <span className="latency__dot" aria-hidden="true" />
      <span className="latency__label">{backendLabel}</span>
      <span className="latency__value" aria-live="polite">
        {latencyMs === null ? "—" : `${latencyMs.toFixed(latencyMs < 10 ? 1 : 0)} ms`}
      </span>
    </div>
  );
}

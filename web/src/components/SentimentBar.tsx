/**
 * The signal meter: a horizontal bar that fills toward the predicted sentiment
 * (green = positive, red = negative) with a smooth CSS transition, plus the
 * verdict word and confidence readout. Shows a skeleton state before the first
 * result instead of a spinner.
 */
import type { ClassifyResult } from "../types";

interface SentimentBarProps {
  result: ClassifyResult | null;
  /** True before the model is ready / first inference completes. */
  skeleton: boolean;
}

export function SentimentBar({ result, skeleton }: SentimentBarProps) {
  const label = result?.label ?? "negative";
  const confidence = result?.confidence ?? 0.5;
  const pct = Math.round(confidence * 100);

  return (
    <div className="meter" data-skeleton={skeleton} data-label={result ? label : "idle"}>
      <div className="meter__head">
        <span className="meter__verdict">
          {skeleton ? "—" : result ? label : "awaiting input"}
        </span>
        <span className="meter__confidence" aria-live="polite">
          {skeleton || !result ? "··" : `${pct}%`}
        </span>
      </div>

      <div
        className="meter__track"
        role="meter"
        aria-valuenow={result ? pct : 0}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Sentiment confidence"
      >
        <span className="meter__ticks" aria-hidden="true" />
        <span
          className="meter__fill"
          data-label={result ? label : "idle"}
          style={{ width: skeleton ? "38%" : `${result ? pct : 50}%` }}
        />
      </div>

      <div className="meter__scale" aria-hidden="true">
        <span>negative</span>
        <span>positive</span>
      </div>
    </div>
  );
}

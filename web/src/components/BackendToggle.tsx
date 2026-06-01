/**
 * Segmented toggle between the two inference backends. Styled as an instrument
 * switch: a sliding "thumb" tracks the active segment.
 */
import type { Backend } from "../types";
import { BACKEND_LABELS } from "../types";

interface BackendToggleProps {
  value: Backend;
  onChange: (backend: Backend) => void;
  disabled?: boolean;
}

const ORDER: Backend[] = ["wasm", "edge"];

export function BackendToggle({ value, onChange, disabled = false }: BackendToggleProps) {
  const activeIndex = ORDER.indexOf(value);

  return (
    <div
      className="backend-toggle"
      role="tablist"
      aria-label="Inference backend"
      data-disabled={disabled}
    >
      <span
        className="backend-toggle__thumb"
        style={{ transform: `translateX(${activeIndex * 100}%)` }}
        aria-hidden="true"
      />
      {ORDER.map((backend) => (
        <button
          key={backend}
          type="button"
          role="tab"
          aria-selected={value === backend}
          className="backend-toggle__option"
          data-active={value === backend}
          disabled={disabled}
          onClick={() => onChange(backend)}
        >
          <span className="backend-toggle__dot" data-backend={backend} aria-hidden="true" />
          {BACKEND_LABELS[backend]}
        </button>
      ))}
    </div>
  );
}

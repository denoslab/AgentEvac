import { useConsole } from '../state/store'
import { StatusDot } from '../ui/primitives'

const ENTRIES = [
  { shape: 'circle', tone: 'neutral', label: 'At home, unaware' },
  { shape: 'circle', tone: 'caution', label: 'At home, warned' },
  { shape: 'triangle', tone: 'moving', label: 'Evacuating' },
  { shape: 'square', tone: 'nominal', label: 'Arrived' },
  { shape: 'flame', tone: 'hazard', label: 'Reached by fire' },
] as const

const LAYERS = [
  { key: 'households', label: 'Households' },
  { key: 'fires', label: 'Fire' },
  { key: 'areas', label: 'Alert areas' },
  { key: 'roads', label: 'Roads' },
  { key: 'destinations', label: 'Shelters' },
] as const

/** The key and the layer switches, always on screen so nothing has to be recalled. */
export function MapLegend() {
  const layers = useConsole((s) => s.layers)
  const toggleLayer = useConsole((s) => s.toggleLayer)

  return (
    <div className="pointer-events-auto absolute bottom-3 left-3 w-56 rounded-panel border border-ink-line bg-ink-panel/95 p-2.5">
      <p className="mb-1.5 text-micro font-semibold uppercase tracking-wide text-ink-muted">Key</p>
      <ul className="space-y-1">
        {ENTRIES.map((entry) => (
          <li key={entry.label} className="flex items-center gap-2 text-micro text-ink-text">
            <StatusDot shape={entry.shape} tone={entry.tone} />
            {entry.label}
          </li>
        ))}
        <li className="flex items-center gap-2 text-micro text-ink-text">
          <span className="h-2.5 w-2.5 rounded-sm border border-status-hazard bg-status-hazard/25" />
          Fire front
        </li>
      </ul>
      <p className="mb-1.5 mt-3 text-micro font-semibold uppercase tracking-wide text-ink-muted">Layers</p>
      <div className="flex flex-wrap gap-1">
        {LAYERS.map((layer) => (
          <button
            key={layer.key}
            type="button"
            onClick={() => toggleLayer(layer.key)}
            aria-pressed={layers[layer.key] !== false}
            className={`rounded border px-1.5 py-0.5 text-micro transition-colors ${
              layers[layer.key] === false
                ? 'border-ink-line text-ink-faint'
                : 'border-status-moving/50 bg-status-moving/12 text-ink-text'
            }`}
          >
            {layer.label}
          </button>
        ))}
      </div>
    </div>
  )
}

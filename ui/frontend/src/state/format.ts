// Formatting shared by every readout. Numbers an operator reads under pressure
// are always rendered through here, so they never disagree between panels.

/** Simulation seconds as h:mm:ss on the incident clock. */
export function simClock(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return '--:--:--'
  const total = Math.max(0, Math.floor(seconds))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

/** Simulation seconds as local wall-clock time, anchored to the incident start. */
export function anchoredClock(seconds: number | null | undefined, anchor: string | null): string | null {
  if (seconds == null || !anchor) return null
  const parts = anchor.split(':').map(Number)
  if (parts.length < 2 || parts.some((n) => !Number.isFinite(n))) return null
  const base = parts[0] * 3600 + parts[1] * 60 + (parts[2] ?? 0)
  const total = Math.floor(base + Math.max(0, seconds))
  const h = Math.floor(total / 3600) % 24
  const m = Math.floor((total % 3600) / 60)
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`
}

/** Compact duration for schedules and outcome tiles. */
export function duration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return '--'
  const total = Math.round(Math.abs(seconds))
  const sign = seconds < 0 ? '-' : ''
  if (total < 60) return `${sign}${total} s`
  const m = Math.floor(total / 60)
  if (m < 60) return `${sign}${m} min`
  const h = Math.floor(m / 60)
  const rest = m % 60
  return rest ? `${sign}${h} h ${rest} min` : `${sign}${h} h`
}

export function minutes(seconds: number | null | undefined, digits = 0): string {
  if (seconds == null || !Number.isFinite(seconds)) return '--'
  return `${(seconds / 60).toFixed(digits)}`
}

export function integer(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '--'
  return Math.round(value).toLocaleString('en-CA')
}

export function decimal(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return '--'
  return value.toFixed(digits)
}

export function percent(value: number | null | undefined, digits = 0): string {
  if (value == null || !Number.isFinite(value)) return '--'
  return `${(value * 100).toFixed(digits)}%`
}

export function speedLabel(speed: number | null | undefined): string {
  if (speed == null) return '--'
  if (speed <= 0) return 'max'
  return `${speed}x`
}

/** Wall-clock timestamp for run history rows. */
export function wallDate(epochSeconds: number | null | undefined): string {
  if (!epochSeconds) return '--'
  const date = new Date(epochSeconds * 1000)
  return date.toLocaleString('en-CA', {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function titleCase(value: string): string {
  return value.replace(/[_-]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

import { type ReactNode, useEffect, useRef, useState } from 'react'

// Shared building blocks. Status is never carried by colour alone: every element
// here pairs its hue with a word, a shape, or an icon.

export type Tone = 'neutral' | 'nominal' | 'caution' | 'hazard' | 'moving'

const TONE_TEXT: Record<Tone, string> = {
  neutral: 'text-ink-text',
  nominal: 'text-status-nominal',
  caution: 'text-status-caution',
  hazard: 'text-status-hazard',
  moving: 'text-status-moving',
}

const TONE_BORDER: Record<Tone, string> = {
  neutral: 'border-ink-line',
  nominal: 'border-status-nominal/50',
  caution: 'border-status-caution/50',
  hazard: 'border-status-hazard/70',
  moving: 'border-status-moving/50',
}

const TONE_FILL: Record<Tone, string> = {
  neutral: 'bg-ink-raised',
  nominal: 'bg-status-nominal/12',
  caution: 'bg-status-caution/12',
  hazard: 'bg-status-hazard/15',
  moving: 'bg-status-moving/12',
}

export function Panel({
  title,
  action,
  children,
  className = '',
  bodyClassName = '',
}: {
  title?: string
  action?: ReactNode
  children: ReactNode
  className?: string
  bodyClassName?: string
}) {
  return (
    // `shrink-0` keeps a panel at its natural height inside a scrolling column.
    // Without it the column shrinks its panels to fit, and their contents spill
    // over the panel below instead of the column scrolling.
    <section className={`panel flex min-h-0 shrink-0 flex-col ${className}`}>
      {title && (
        <header className="panel-title flex items-center justify-between gap-2">
          <span>{title}</span>
          {action}
        </header>
      )}
      <div className={`min-h-0 flex-1 ${bodyClassName || 'p-4'}`}>{children}</div>
    </section>
  )
}

export function StatTile({
  label,
  value,
  hint,
  tone = 'neutral',
  large = false,
  shape,
}: {
  label: string
  value: ReactNode
  hint?: ReactNode
  tone?: Tone
  large?: boolean
  shape?: StatusShape
}) {
  return (
    <div className={`rounded-panel border px-3 py-2.5 ${TONE_BORDER[tone]} ${TONE_FILL[tone]}`}>
      <div className="flex items-center gap-1.5">
        {shape && <StatusDot shape={shape} tone={tone} />}
        <span className="text-micro font-semibold uppercase tracking-wide text-ink-muted">{label}</span>
      </div>
      <div className={`tnum mt-1 font-semibold ${large ? 'text-readout' : 'text-view'} ${TONE_TEXT[tone]}`}>
        {value}
      </div>
      {hint && <div className="mt-0.5 text-micro text-ink-faint">{hint}</div>}
    </div>
  )
}

export type StatusShape = 'circle' | 'triangle' | 'square' | 'flame' | 'bar'

/**
 * A status marker whose shape repeats what its colour says, so the meaning
 * survives a projector, a photocopy, and colour vision deficiency alike.
 */
export function StatusDot({ shape, tone = 'neutral' }: { shape: StatusShape; tone?: Tone }) {
  const fill = {
    neutral: '#8D99A6',
    nominal: '#009E73',
    caution: '#E69F00',
    hazard: '#D55E00',
    moving: '#0072B2',
  }[tone]
  return (
    <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden className="shrink-0">
      {shape === 'circle' && <circle cx="5" cy="5" r="4" fill={fill} />}
      {shape === 'triangle' && <path d="M5 1 L9 8.5 L1 8.5 Z" fill={fill} />}
      {shape === 'square' && <rect x="1.5" y="1.5" width="7" height="7" fill={fill} />}
      {shape === 'flame' && <path d="M5 0.5 C7 3 8.5 4 8.5 6.2 A3.5 3.5 0 0 1 1.5 6.2 C1.5 4.4 3 3.4 3.6 1.8 C4.2 3 4.4 3.4 5 0.5 Z" fill={fill} />}
      {shape === 'bar' && <rect x="0.5" y="3.5" width="9" height="3" fill={fill} />}
    </svg>
  )
}

export function Badge({
  children,
  tone = 'neutral',
  shape,
}: {
  children: ReactNode
  tone?: Tone
  shape?: StatusShape
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded border px-1.5 py-0.5 text-micro font-medium ${TONE_BORDER[tone]} ${TONE_FILL[tone]} ${TONE_TEXT[tone]}`}
    >
      {shape && <StatusDot shape={shape} tone={tone} />}
      {children}
    </span>
  )
}

export function EmptyState({
  title,
  detail,
  action,
}: {
  title: string
  detail?: string
  action?: ReactNode
}) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center">
      <p className="text-base font-medium text-ink-muted">{title}</p>
      {detail && <p className="max-w-sm text-small text-ink-faint">{detail}</p>}
      {action}
    </div>
  )
}

/**
 * A button that carries the reason it is disabled, so a blocked control never
 * leaves an operator guessing.
 */
export function Button({
  children,
  onClick,
  variant = 'default',
  disabled = false,
  disabledReason,
  busy = false,
  className = '',
  type = 'button',
  title,
}: {
  children: ReactNode
  onClick?: () => void
  variant?: 'default' | 'primary' | 'danger' | 'ghost'
  disabled?: boolean
  disabledReason?: string
  busy?: boolean
  className?: string
  type?: 'button' | 'submit'
  title?: string
}) {
  const variantClass = {
    default: '',
    primary: 'btn-primary',
    danger: 'btn-danger',
    ghost: 'btn-ghost',
  }[variant]
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || busy}
      title={disabled && disabledReason ? disabledReason : title}
      aria-disabled={disabled || busy}
      className={`btn ${variantClass} ${className}`}
    >
      {busy && <Spinner />}
      {children}
    </button>
  )
}

export function Spinner() {
  return (
    <svg className="h-3.5 w-3.5 animate-spin" viewBox="0 0 16 16" aria-hidden>
      <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeOpacity="0.25" strokeWidth="2" />
      <path d="M14 8a6 6 0 0 0-6-6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  )
}

export function ProgressBar({
  value,
  max,
  tone = 'moving',
  height = 6,
}: {
  value: number
  max: number
  tone?: Tone
  height?: number
}) {
  const fraction = max > 0 ? Math.min(1, Math.max(0, value / max)) : 0
  const fill = {
    neutral: '#8D99A6',
    nominal: '#009E73',
    caution: '#E69F00',
    hazard: '#D55E00',
    moving: '#0072B2',
  }[tone]
  return (
    <div className="w-full overflow-hidden rounded-full bg-ink-raised" style={{ height }}>
      <div
        className="h-full rounded-full transition-[width] duration-200"
        style={{ width: `${fraction * 100}%`, background: fill }}
      />
    </div>
  )
}

/**
 * A panel that folds away, keeping its heading and a one-line summary visible.
 *
 * Pass `open` and `onOpenChange` to drive it from outside, for an expand-all
 * control. Left uncontrolled it remembers its own state from `defaultOpen`.
 */
export function Collapsible({
  title,
  children,
  defaultOpen = false,
  meta,
  open: openProp,
  onOpenChange,
  className = '',
  bodyClassName = 'p-4',
}: {
  title: string
  children: ReactNode
  defaultOpen?: boolean
  meta?: ReactNode
  open?: boolean
  onOpenChange?: (open: boolean) => void
  className?: string
  bodyClassName?: string
}) {
  const [uncontrolled, setUncontrolled] = useState(defaultOpen)
  const open = openProp ?? uncontrolled
  const toggle = () => {
    if (openProp === undefined) setUncontrolled(!open)
    onOpenChange?.(!open)
  }
  return (
    <div className={`panel shrink-0 ${className}`}>
      <button
        type="button"
        onClick={toggle}
        className="flex w-full items-center justify-between gap-2 px-4 py-2.5 text-left hover:bg-ink-raised/40"
        aria-expanded={open}
      >
        <span className="text-small font-semibold uppercase tracking-wide text-ink-muted">{title}</span>
        <span className="flex min-w-0 items-center gap-2 text-micro text-ink-faint">
          {/* A folded section still says what it holds, so nothing is hidden in
              substance by being hidden from view. */}
          <span className="truncate">{meta}</span>
          <svg
            width="12"
            height="12"
            viewBox="0 0 12 12"
            aria-hidden
            className={`shrink-0 transition-transform ${open ? 'rotate-90' : ''}`}
          >
            <path d="M4 2 L8 6 L4 10" fill="none" stroke="currentColor" strokeWidth="1.6" />
          </svg>
        </span>
      </button>
      {open && <div className={`border-t border-ink-line ${bodyClassName}`}>{children}</div>}
    </div>
  )
}

/** A destructive action always states its consequence before it runs. */
export function ConfirmModal({
  open,
  title,
  consequence,
  confirmLabel,
  onConfirm,
  onCancel,
}: {
  open: boolean
  title: string
  consequence: string
  confirmLabel: string
  onConfirm: () => void
  onCancel: () => void
}) {
  const confirmRef = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    if (!open) return
    confirmRef.current?.focus()
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onCancel()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onCancel])

  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" role="dialog" aria-modal>
      <div className="panel w-full max-w-md p-5">
        <h2 className="text-panel font-semibold">{title}</h2>
        <p className="mt-2 text-base text-ink-muted">{consequence}</p>
        <div className="mt-5 flex justify-end gap-2">
          <Button onClick={onCancel} variant="ghost">
            Keep running
          </Button>
          <button ref={confirmRef} type="button" onClick={onConfirm} className="btn btn-danger">
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}

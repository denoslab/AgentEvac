import { useEffect } from 'react'
import { useConsole, type Toast } from '../state/store'

const TONE_CLASS: Record<Toast['tone'], string> = {
  info: 'border-ink-line bg-ink-raised text-ink-text',
  good: 'border-status-nominal/60 bg-status-nominal/15 text-status-nominal',
  warn: 'border-status-caution/60 bg-status-caution/15 text-status-caution',
  bad: 'border-status-hazard/70 bg-status-hazard/20 text-status-hazard',
}

function ToastRow({ toast }: { toast: Toast }) {
  const dismiss = useConsole((s) => s.dismissToast)
  useEffect(() => {
    const timer = window.setTimeout(() => dismiss(toast.id), toast.action ? 12000 : 6000)
    return () => window.clearTimeout(timer)
  }, [toast, dismiss])

  return (
    <div className={`flex items-center gap-3 rounded-panel border px-3 py-2 text-small shadow-lg ${TONE_CLASS[toast.tone]}`}>
      <span className="flex-1">{toast.message}</span>
      {toast.action && (
        <button
          type="button"
          className="underline underline-offset-2"
          onClick={() => {
            toast.action?.run()
            dismiss(toast.id)
          }}
        >
          {toast.action.label}
        </button>
      )}
      <button type="button" aria-label="Dismiss" className="text-current opacity-60 hover:opacity-100" onClick={() => dismiss(toast.id)}>
        ×
      </button>
    </div>
  )
}

/** Command acknowledgements and notices, announced to assistive technology. */
export function Toasts() {
  const toasts = useConsole((s) => s.toasts)
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-40 flex w-96 flex-col gap-2" aria-live="polite">
      {toasts.map((toast) => (
        <div key={toast.id} className="pointer-events-auto">
          <ToastRow toast={toast} />
        </div>
      ))}
    </div>
  )
}

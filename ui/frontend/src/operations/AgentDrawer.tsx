import { useEffect, useState } from 'react'
import { api } from '../state/api'
import { decimal, simClock, titleCase } from '../state/format'
import { useConsole } from '../state/store'
import { Badge, Spinner } from '../ui/primitives'

interface AgentDetail {
  agent_id: string
  mode?: string
  belief?: Record<string, number>
  psychology?: Record<string, unknown>
  profile?: Record<string, number>
  current?: Record<string, unknown>
  latest?: Record<string, unknown>
  histories?: Record<string, unknown[]>
  inbox?: unknown[]
  error?: string
}

function BeliefBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-14 shrink-0 text-micro text-ink-muted">{label}</span>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-ink-raised">
        <div className="h-full rounded-full" style={{ width: `${Math.min(1, value) * 100}%`, background: color }} />
      </div>
      <span className="tnum w-10 text-right text-micro text-ink-text">{decimal(value, 2)}</span>
    </div>
  )
}

function Row({ label, value }: { label: string; value: unknown }) {
  if (value == null || value === '') return null
  const text = typeof value === 'number' ? decimal(value, 3) : String(value)
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-ink-line/60 py-1 last:border-0">
      <span className="text-micro text-ink-muted">{titleCase(label)}</span>
      <span className="tnum truncate text-micro text-ink-text" title={text}>
        {text}
      </span>
    </div>
  )
}

/** One household's state, opened by clicking it on the map or in the log. */
export function AgentDrawer() {
  const selected = useConsole((s) => s.selectedAgent)
  const selectAgent = useConsole((s) => s.selectAgent)
  const snapshot = useConsole((s) => s.snapshot)
  const events = useConsole((s) => s.events)
  const [detail, setDetail] = useState<AgentDetail | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!selected) {
      setDetail(null)
      return
    }
    let cancelled = false
    setLoading(true)
    api
      .agentDetail(selected)
      .then((data) => {
        if (!cancelled) setDetail(data as AgentDetail)
      })
      .catch(() => {
        if (!cancelled) setDetail({ agent_id: selected, error: 'This household is not reachable right now.' })
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [selected])

  if (!selected) return null

  const frame = snapshot?.agents.find((agent) => agent.id === selected) ?? null
  const own = events.filter((event) => event.veh_id === selected || event.agent_id === selected).slice(-25).reverse()
  const belief = detail?.belief ?? {}

  return (
    <aside className="absolute right-0 top-0 z-20 flex h-full w-[340px] flex-col border-l border-ink-line bg-ink-panel shadow-2xl">
      <header className="flex items-center justify-between gap-2 border-b border-ink-line px-4 py-2.5">
        <div className="min-w-0">
          <p className="truncate font-mono text-small font-semibold">{selected}</p>
          <p className="text-micro text-ink-faint">{detail?.mode ? titleCase(detail.mode) : 'household'}</p>
        </div>
        <button type="button" onClick={() => selectAgent(null)} className="btn btn-ghost h-7 px-2" aria-label="Close">
          ×
        </button>
      </header>

      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
        <div className="flex flex-wrap gap-1.5">
          {frame && (
            <Badge
              tone={frame.status === 'evacuating' ? 'moving' : frame.status === 'arrived' ? 'nominal' : 'neutral'}
              shape={frame.status === 'evacuating' ? 'triangle' : frame.status === 'arrived' ? 'square' : 'circle'}
            >
              {titleCase(frame.status)}
            </Badge>
          )}
          {frame?.aware && <Badge tone="caution">Warned</Badge>}
          {frame?.fire_contact && (
            <Badge tone="hazard" shape="flame">
              Reached by fire
            </Badge>
          )}
        </div>

        {loading && (
          <p className="flex items-center gap-2 text-micro text-ink-muted">
            <Spinner /> reading state from the run
          </p>
        )}
        {detail?.error && <p className="text-micro text-status-caution">{detail.error}</p>}

        {Object.keys(belief).length > 0 && (
          <section>
            <h3 className="mb-2 text-micro font-semibold uppercase tracking-wide text-ink-muted">Belief</h3>
            <div className="space-y-1.5">
              <BeliefBar label="safe" value={Number(belief.p_safe ?? 0)} color="#009E73" />
              <BeliefBar label="risky" value={Number(belief.p_risky ?? 0)} color="#E69F00" />
              <BeliefBar label="danger" value={Number(belief.p_danger ?? 0)} color="#D55E00" />
            </div>
          </section>
        )}

        {detail?.profile && Object.keys(detail.profile).length > 0 && (
          <section>
            <h3 className="mb-1 text-micro font-semibold uppercase tracking-wide text-ink-muted">Profile</h3>
            {Object.entries(detail.profile).map(([key, value]) => (
              <Row key={key} label={key} value={value} />
            ))}
          </section>
        )}

        {detail?.current && Object.keys(detail.current).length > 0 && (
          <section>
            <h3 className="mb-1 text-micro font-semibold uppercase tracking-wide text-ink-muted">Position</h3>
            {Object.entries(detail.current).map(([key, value]) =>
              typeof value === 'object' ? null : <Row key={key} label={key} value={value} />,
            )}
          </section>
        )}

        <section>
          <h3 className="mb-1 text-micro font-semibold uppercase tracking-wide text-ink-muted">
            What this household did
          </h3>
          {own.length === 0 ? (
            <p className="text-micro text-ink-faint">No events recorded for this household yet.</p>
          ) : (
            <ul className="space-y-1">
              {own.map((event) => (
                <li key={event.seq} className="flex gap-2 text-micro">
                  <span className="tnum w-14 shrink-0 text-ink-faint">
                    {typeof event.sim_t_s === 'number' ? simClock(event.sim_t_s) : '--'}
                  </span>
                  <span className="min-w-0 flex-1 text-ink-text">{event.summary ?? String(event.event)}</span>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </aside>
  )
}

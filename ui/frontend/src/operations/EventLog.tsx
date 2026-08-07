import { useEffect, useMemo, useRef } from 'react'
import { simClock } from '../state/format'
import { useConsole } from '../state/store'
import type { SimEvent } from '../state/types'
import { Badge, EmptyState } from '../ui/primitives'

// Event families the tabs group by. A family the simulator adds later still
// shows up under "all", so the log never silently drops a record.
const FAMILIES: Record<string, string[]> = {
  alerts: ['alert_issued', 'awareness', 'door_knock'],
  fire: ['ignition', 'fire_contact', 'fire_reached'],
  movement: ['departure_release', 'arrival', 'route_applied', 'route_apply_error', 'route_skip'],
  decisions: [
    'decision_round_start',
    'llm_decision',
    'llm_error',
    'predeparture_llm_decision',
    'predeparture_llm_error',
    'replay_apply_round',
  ],
  messages: ['message_queued', 'message_delivered'],
}

const TABS = [
  { id: 'all', label: 'All' },
  { id: 'alerts', label: 'Alerts' },
  { id: 'fire', label: 'Fire' },
  { id: 'movement', label: 'Movement' },
  { id: 'decisions', label: 'Decisions' },
  { id: 'messages', label: 'Household messages' },
]

function toneFor(event: string): 'neutral' | 'nominal' | 'caution' | 'hazard' | 'moving' {
  if (event.endsWith('_error') || event === 'ui_bridge_warning') return 'caution'
  if (FAMILIES.fire.includes(event)) return 'hazard'
  if (FAMILIES.alerts.includes(event)) return 'caution'
  if (event === 'arrival') return 'nominal'
  if (FAMILIES.movement.includes(event)) return 'moving'
  return 'neutral'
}

function agentOf(event: SimEvent): string | null {
  const candidate = event.veh_id ?? event.agent_id ?? event.sender ?? event.recipient
  return typeof candidate === 'string' ? candidate : null
}

function messageText(event: SimEvent): string | null {
  const text = event.text ?? event.message ?? event.body
  return typeof text === 'string' ? text : null
}

export function EventLog() {
  const events = useConsole((s) => s.events)
  const filter = useConsole((s) => s.eventFilter)
  const setFilter = useConsole((s) => s.setEventFilter)
  const follow = useConsole((s) => s.followLatest)
  const setFollow = useConsole((s) => s.setFollowLatest)
  const selectAgent = useConsole((s) => s.selectAgent)
  const phase = useConsole((s) => s.session.phase)
  const scroller = useRef<HTMLDivElement>(null)

  const visible = useMemo(() => {
    if (filter === 'all') return events
    const allowed = new Set(FAMILIES[filter] ?? [])
    return events.filter((event) => allowed.has(String(event.event)))
  }, [events, filter])

  useEffect(() => {
    if (!follow || !scroller.current) return
    scroller.current.scrollTop = scroller.current.scrollHeight
  }, [visible.length, follow])

  const onScroll = () => {
    const node = scroller.current
    if (!node) return
    const atBottom = node.scrollHeight - node.scrollTop - node.clientHeight < 40
    if (atBottom !== follow) setFollow(atBottom)
  }

  const counts = useMemo(() => {
    const out: Record<string, number> = { all: events.length }
    for (const [family, names] of Object.entries(FAMILIES)) {
      const allowed = new Set(names)
      out[family] = events.filter((event) => allowed.has(String(event.event))).length
    }
    return out
  }, [events])

  return (
    <section className="panel flex min-h-0 flex-col">
      <header className="flex shrink-0 items-center gap-1 border-b border-ink-line px-2 py-1.5">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setFilter(tab.id)}
            aria-pressed={filter === tab.id}
            className={`rounded px-2 py-1 text-micro font-medium transition-colors ${
              filter === tab.id ? 'bg-ink-raised text-ink-text' : 'text-ink-muted hover:bg-ink-raised/60'
            }`}
          >
            {tab.label}
            <span className="tnum ml-1.5 text-ink-faint">{counts[tab.id] ?? 0}</span>
          </button>
        ))}
        {!follow && (
          <button
            type="button"
            onClick={() => {
              setFollow(true)
              if (scroller.current) scroller.current.scrollTop = scroller.current.scrollHeight
            }}
            className="ml-auto rounded border border-status-caution/50 px-2 py-1 text-micro text-status-caution"
          >
            Jump to latest (L)
          </button>
        )}
      </header>

      <div ref={scroller} onScroll={onScroll} className="min-h-0 flex-1 overflow-y-auto px-2 py-1">
        {visible.length === 0 ? (
          <EmptyState
            title={phase === 'idle' ? 'No run has been started' : 'Waiting for the first event'}
            detail="Alerts, ignitions, departures, arrivals, and every household decision appear here as they happen."
          />
        ) : (
          <ul className="space-y-0.5">
            {visible.map((event) => {
              const agent = agentOf(event)
              const text = filter === 'messages' ? messageText(event) : null
              return (
                <li
                  key={event.seq}
                  className={`flex items-start gap-2 rounded px-1.5 py-1 text-micro ${
                    agent ? 'cursor-pointer hover:bg-ink-raised' : ''
                  }`}
                  onClick={() => agent && selectAgent(agent)}
                >
                  <span className="tnum w-16 shrink-0 text-ink-faint">
                    {typeof event.sim_t_s === 'number' ? simClock(event.sim_t_s) : '--'}
                  </span>
                  <Badge tone={toneFor(String(event.event))}>{String(event.event).replace(/_/g, ' ')}</Badge>
                  <span className="min-w-0 flex-1 text-ink-text">
                    {text ? <span className="italic text-ink-muted">“{text}”</span> : (event.summary ?? '')}
                  </span>
                </li>
              )
            })}
          </ul>
        )}
      </div>
    </section>
  )
}

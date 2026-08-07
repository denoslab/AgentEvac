import { useState } from 'react'
import { duration } from '../state/format'
import { useConsole } from '../state/store'
import { LiveMap } from '../map/LiveMap'
import { MapLegend } from '../map/MapLegend'
import { Button, EmptyState, Spinner } from '../ui/primitives'
import { AgentDrawer } from './AgentDrawer'
import { EventLog } from './EventLog'
import { KpiRail, RoundBanner } from './panels'

/** Loading a metropolitan road network takes real time, so it is shown as work. */
function PreparingOverlay() {
  const session = useConsole((s) => s.session)
  if (session.phase !== 'preparing') return null
  return (
    <div className="absolute inset-0 z-10 flex items-center justify-center bg-ink-bg/80">
      <div className="panel max-w-md p-6 text-center">
        <p className="flex items-center justify-center gap-2 text-panel font-semibold">
          <Spinner /> Preparing the run
        </p>
        <p className="mt-2 text-base text-ink-muted">{session.detail}</p>
        {session.elapsed_wall_s != null && (
          <p className="tnum mt-1 text-small text-ink-faint">{duration(session.elapsed_wall_s)} elapsed</p>
        )}
        <p className="mt-3 text-micro text-ink-faint">
          The Halifax network is a 240 MB file, so this stage takes tens of seconds before the first vehicle moves.
        </p>
      </div>
    </div>
  )
}

function FailurePanel() {
  const session = useConsole((s) => s.session)
  const setView = useConsole((s) => s.setView)
  if (session.phase !== 'failed') return null
  return (
    <div className="absolute inset-0 z-10 flex items-center justify-center bg-ink-bg/90 p-6">
      <div className="panel max-h-full w-full max-w-2xl overflow-auto p-6">
        <h2 className="text-view font-semibold text-status-hazard">The run stopped with an error</h2>
        <p className="mt-2 text-base text-ink-muted">{session.error ?? session.detail}</p>
        {session.stderr_tail.length > 0 && (
          <pre className="mt-4 max-h-64 overflow-auto rounded-panel border border-ink-line bg-ink-bg p-3 font-mono text-micro text-ink-muted">
            {session.stderr_tail.join('\n')}
          </pre>
        )}
        <div className="mt-5 flex gap-2">
          <Button variant="primary" onClick={() => setView('setup')}>
            Back to Setup
          </Button>
          <Button onClick={() => setView('debrief')}>Open whatever was exported</Button>
        </div>
      </div>
    </div>
  )
}

export function OperationsView() {
  const phase = useConsole((s) => s.session.phase)
  const setView = useConsole((s) => s.setView)
  const [logOpen, setLogOpen] = useState(true)

  if (phase === 'idle') {
    return (
      <EmptyState
        title="No run is in flight"
        detail="Configure a scenario in Setup and launch it. This view then shows the incident as it unfolds."
        action={
          <Button variant="primary" onClick={() => setView('setup')} className="mt-2">
            Go to Setup
          </Button>
        }
      />
    )
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <RoundBanner />
      <div className="relative flex min-h-0 flex-1 flex-col gap-3 p-3">
        <div className="flex min-h-0 flex-1 gap-3">
          <div className="relative min-w-0 flex-1">
            <LiveMap />
            <MapLegend />
            <AgentDrawer />
          </div>
          <KpiRail />
        </div>

        <div className={`flex shrink-0 flex-col ${logOpen ? 'h-[200px]' : 'h-[34px]'}`}>
          <button
            type="button"
            onClick={() => setLogOpen((open) => !open)}
            className="flex items-center gap-2 self-start rounded-t-panel px-2 py-1 text-micro text-ink-muted hover:text-ink-text"
            aria-expanded={logOpen}
          >
            <svg width="10" height="10" viewBox="0 0 12 12" aria-hidden className={logOpen ? '' : 'rotate-180'}>
              <path d="M2 8 L6 4 L10 8" fill="none" stroke="currentColor" strokeWidth="1.6" />
            </svg>
            Event log
          </button>
          {logOpen && (
            <div className="min-h-0 flex-1">
              <EventLog />
            </div>
          )}
        </div>

        <PreparingOverlay />
        <FailurePanel />
      </div>
    </div>
  )
}

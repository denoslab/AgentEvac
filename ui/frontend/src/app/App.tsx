import { useEffect, useState } from 'react'
import { api } from '../state/api'
import { simClock } from '../state/format'
import { connectStream, disconnectStream } from '../state/stream'
import { useConsole } from '../state/store'
import type { OrphanRun } from '../state/types'
import { DebriefView } from '../debrief/DebriefView'
import { OperationsView } from '../operations/OperationsView'
import { AuthorView } from '../authoring/AuthorView'
import { SetupView } from '../setup/SetupView'
import { Button, EmptyState } from '../ui/primitives'
import { Toasts } from '../ui/Toasts'
import { TopBar } from './TopBar'

/**
 * Keyboard shortcuts for the controls an operator reaches for mid-run. They stay
 * out of the way while a text field has focus.
 */
function useShortcuts() {
  const setFollow = useConsole((s) => s.setFollowLatest)
  const pushToast = useConsole((s) => s.pushToast)

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return
      if (event.metaKey || event.ctrlKey || event.altKey) return

      const phase = useConsole.getState().session.phase
      const controllable = phase === 'running' || phase === 'paused'

      if (event.code === 'Space' && controllable) {
        event.preventDefault()
        void api.control('toggle_pause').catch(() => pushToast('pause failed', 'bad'))
        return
      }
      if (event.key >= '1' && event.key <= '5' && controllable) {
        const speeds = [1, 4, 16, 60, 0]
        void api
          .control('set_speed', speeds[Number(event.key) - 1])
          .catch(() => pushToast('speed change failed', 'bad'))
        return
      }
      if (event.key.toLowerCase() === 'l') setFollow(true)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [setFollow, pushToast])
}

/** A backend that cannot be reached gets a labelled screen, never a blank one. */
function BackendDown({ onRetry }: { onRetry: () => void }) {
  return (
    <div className="flex h-full items-center justify-center">
      <EmptyState
        title="The console backend is not answering"
        detail="Start it with: python -m ui.backend --port 8000, then reconnect."
        action={
          <Button variant="primary" onClick={onRetry} className="mt-2">
            Reconnect
          </Button>
        }
      />
    </div>
  )
}

/**
 * A simulation that outlived the backend is still burning wall time and holding
 * its port. The operator is told, and given both ways out.
 */
function OrphanPrompt({ orphan, onResolved }: { orphan: OrphanRun; onResolved: () => void }) {
  const setSession = useConsole((s) => s.setSession)
  const pushToast = useConsole((s) => s.pushToast)
  const [busy, setBusy] = useState<string | null>(null)

  const act = async (choice: 'adopt' | 'discard') => {
    setBusy(choice)
    try {
      if (choice === 'adopt') {
        const result = await api.adoptOrphan()
        setSession(result.session)
        useConsole.getState().setView('operations')
      } else {
        await api.discardOrphan()
        pushToast('The earlier run was told to stop and export what it had.', 'info')
      }
      onResolved()
    } catch {
      pushToast('That run could not be reached. It may have already stopped.', 'warn')
      onResolved()
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="panel w-full max-w-lg p-6">
        <h2 className="text-view font-semibold text-status-caution">A run is still going</h2>
        <p className="mt-2 text-base text-ink-muted">
          The console restarted while a simulation was in flight. It is still running at{' '}
          <span className="tnum">{simClock(orphan.sim_t_s)}</span> into the incident.
        </p>
        {orphan.label && <p className="mt-1 text-small text-ink-faint">{orphan.label}</p>}
        <div className="mt-5 flex flex-wrap justify-end gap-2">
          <Button onClick={() => act('discard')} variant="danger" busy={busy === 'discard'}>
            Stop it and export
          </Button>
          <Button onClick={() => act('adopt')} variant="primary" busy={busy === 'adopt'}>
            Take control of it
          </Button>
        </div>
      </div>
    </div>
  )
}

export function App() {
  const view = useConsole((s) => s.view)
  const connection = useConsole((s) => s.connection)
  const setSession = useConsole((s) => s.setSession)
  const [booted, setBooted] = useState(false)
  const [reachable, setReachable] = useState(true)
  const [orphan, setOrphan] = useState<OrphanRun | null>(null)
  useShortcuts()

  const boot = () => {
    api
      .health()
      .then((health) => setOrphan(health.orphan_run))
      .catch(() => setOrphan(null))
    api
      .currentRun()
      .then((session) => {
        setSession(session)
        setReachable(true)
        // A backend that already has a run in flight hands the operator straight
        // to it, which is what a reloaded browser or a second screen needs.
        if (session.active) useConsole.getState().setView('operations')
        else if (session.phase === 'complete' || session.phase === 'ended_by_operator')
          useConsole.getState().setView('debrief')
      })
      .catch(() => setReachable(false))
      .finally(() => setBooted(true))
    connectStream()
  }

  useEffect(() => {
    boot()
    return () => disconnectStream()
    // Boot runs once. Reconnection is handled by the stream client itself.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  if (!booted) {
    return <div className="flex h-full items-center justify-center text-ink-muted">Starting the console</div>
  }

  if (!reachable && connection !== 'open') {
    return (
      <BackendDown
        onRetry={() => {
          setBooted(false)
          boot()
        }}
      />
    )
  }

  return (
    <div className="flex h-full flex-col">
      <TopBar />
      <main className="min-h-0 flex-1">
        {view === 'setup' && <SetupView />}
        {view === 'author' && (
          <AuthorView onDone={() => useConsole.getState().setView('setup')} />
        )}
        {view === 'operations' && <OperationsView />}
        {view === 'debrief' && <DebriefView />}
      </main>
      {orphan && <OrphanPrompt orphan={orphan} onResolved={() => setOrphan(null)} />}
      <Toasts />
    </div>
  )
}

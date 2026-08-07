import { useEffect, useState } from 'react'
import { api, ApiError } from '../state/api'
import { anchoredClock, simClock, speedLabel } from '../state/format'
import { useConsole, type ViewName } from '../state/store'
import type { SessionPhase } from '../state/types'
import { Button, ConfirmModal, Spinner, type Tone } from '../ui/primitives'

const PHASE_LABEL: Record<SessionPhase, string> = {
  idle: 'No run',
  preparing: 'Preparing',
  running: 'Running',
  paused: 'Paused',
  finishing: 'Finishing',
  complete: 'Complete',
  ended_by_operator: 'Ended',
  failed: 'Failed',
}

const PHASE_TONE: Record<SessionPhase, Tone> = {
  idle: 'neutral',
  preparing: 'caution',
  running: 'nominal',
  paused: 'caution',
  finishing: 'caution',
  complete: 'nominal',
  ended_by_operator: 'nominal',
  failed: 'hazard',
}

const TONE_CLASS: Record<Tone, string> = {
  neutral: 'border-ink-line bg-ink-raised text-ink-muted',
  nominal: 'border-status-nominal/60 bg-status-nominal/15 text-status-nominal',
  caution: 'border-status-caution/60 bg-status-caution/15 text-status-caution',
  hazard: 'border-status-hazard/70 bg-status-hazard/20 text-status-hazard',
  moving: 'border-status-moving/60 bg-status-moving/15 text-status-moving',
}

const SPEEDS = [1, 4, 16, 60, 0]

function RunStatePill() {
  const phase = useConsole((s) => s.session.phase)
  const detail = useConsole((s) => s.session.detail)
  return (
    <div
      className={`flex items-center gap-2 rounded-panel border px-3 py-1.5 ${TONE_CLASS[PHASE_TONE[phase]]}`}
      role="status"
      aria-live="polite"
      title={detail}
    >
      {(phase === 'preparing' || phase === 'finishing') && <Spinner />}
      <span className="text-panel font-semibold uppercase tracking-wide">{PHASE_LABEL[phase]}</span>
    </div>
  )
}

function IncidentClock() {
  const snapshot = useConsole((s) => s.snapshot)
  const anchor = useConsole((s) => s.session.anchor_clock)
  const phase = useConsole((s) => s.session.phase)
  const wall = useConsole((s) => s.snapshotWall)
  const [stalled, setStalled] = useState(false)

  // A clock that has stopped receiving updates must say so rather than quietly
  // showing an old number as if it were current.
  useEffect(() => {
    if (phase !== 'running') {
      setStalled(false)
      return
    }
    const timer = window.setInterval(() => setStalled(performance.now() - wall > 6000), 1000)
    return () => window.clearInterval(timer)
  }, [wall, phase])

  const local = anchoredClock(snapshot?.sim_t_s, anchor)
  return (
    <div className="flex items-baseline gap-3">
      <div className="tnum text-readout font-semibold leading-none">{simClock(snapshot?.sim_t_s)}</div>
      <div className="flex flex-col leading-tight">
        <span className="text-micro uppercase tracking-wide text-ink-faint">into the incident</span>
        {local && <span className="tnum text-small font-medium text-ink-muted">local {local}</span>}
      </div>
      {stalled && (
        <span className="rounded border border-status-caution/60 px-1.5 py-0.5 text-micro text-status-caution">
          no update
        </span>
      )}
    </div>
  )
}

function SpeedSelector({ disabled }: { disabled: boolean }) {
  const snapshot = useConsole((s) => s.snapshot)
  const pushToast = useConsole((s) => s.pushToast)
  const [pending, setPending] = useState<number | null>(null)
  const target = snapshot?.speed_target ?? null
  const inRound = Boolean(snapshot?.round?.in_progress)

  const set = async (value: number) => {
    setPending(value)
    try {
      await api.control('set_speed', value)
    } catch (error) {
      pushToast(error instanceof ApiError ? error.message : 'speed change failed', 'bad')
    } finally {
      setPending(null)
    }
  }

  return (
    <div className="flex items-center gap-2">
      <div className="flex overflow-hidden rounded-panel border border-ink-line" role="group" aria-label="Playback speed">
        {SPEEDS.map((speed) => {
          const active = target === speed
          return (
            <button
              key={speed}
              type="button"
              disabled={disabled}
              onClick={() => set(speed)}
              className={`tnum min-w-[38px] px-2 py-1 text-small font-medium transition-colors disabled:opacity-40 ${
                active ? 'bg-status-moving/25 text-status-moving' : 'text-ink-muted hover:bg-ink-raised'
              }`}
            >
              {pending === speed ? '…' : speedLabel(speed)}
            </button>
          )
        })}
      </div>
      {/* Target next to actual, so a clock frozen by a decision round is not
          mistaken for a hang. */}
      <span className="tnum w-16 text-micro text-ink-faint">
        actual {inRound ? '0x' : speedLabel(target)}
      </span>
    </div>
  )
}

function ConnectionBadge() {
  const connection = useConsole((s) => s.connection)
  const retry = useConsole((s) => s.retryInSeconds)
  const label =
    connection === 'open'
      ? 'Live'
      : connection === 'connecting'
        ? 'Connecting'
        : connection === 'retrying'
          ? `Reconnecting in ${retry}s`
          : 'Offline'
  const tone: Tone = connection === 'open' ? 'nominal' : connection === 'closed' ? 'hazard' : 'caution'
  return (
    <span
      className={`flex items-center gap-1.5 rounded border px-2 py-1 text-micro font-medium ${TONE_CLASS[tone]}`}
      title="Health of the telemetry stream from the console backend"
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {label}
    </span>
  )
}

function RunControls() {
  const phase = useConsole((s) => s.session.phase)
  const pushToast = useConsole((s) => s.pushToast)
  const [busy, setBusy] = useState<string | null>(null)
  const [confirming, setConfirming] = useState(false)

  const controllable = phase === 'running' || phase === 'paused'
  const send = async (action: string) => {
    setBusy(action)
    try {
      await api.control(action)
    } catch (error) {
      pushToast(error instanceof ApiError ? error.message : `${action} failed`, 'bad')
    } finally {
      setBusy(null)
    }
  }

  return (
    <>
      <div className="flex items-center gap-2">
        <Button
          onClick={() => send(phase === 'paused' ? 'resume' : 'pause')}
          disabled={!controllable}
          disabledReason={`The run is ${phase}, so it cannot be paused`}
          busy={busy === 'pause' || busy === 'resume'}
          className="w-[104px]"
        >
          {phase === 'paused' ? 'Resume' : 'Pause'}
        </Button>
        <Button
          onClick={() => setConfirming(true)}
          variant="danger"
          disabled={!controllable}
          disabledReason={`The run is ${phase}, so there is nothing to end`}
          busy={busy === 'end'}
        >
          End run
        </Button>
      </div>
      <ConfirmModal
        open={confirming}
        title="End this run now?"
        consequence="The simulation stops at the current second and exports the results it has so far. Everything already recorded is kept, and the Debrief opens on the partial run."
        confirmLabel="End the run"
        onConfirm={() => {
          setConfirming(false)
          void send('end')
        }}
        onCancel={() => setConfirming(false)}
      />
    </>
  )
}

const VIEWS: { id: ViewName; label: string }[] = [
  { id: 'setup', label: 'Setup' },
  // Authoring sits beside Setup rather than in the run sequence, because it produces a
  // package to run later instead of advancing a run.
  { id: 'author', label: 'Author' },
  { id: 'operations', label: 'Operations' },
  { id: 'debrief', label: 'Debrief' },
]

/**
 * Stopping the console and releasing its port.
 *
 * A run still in flight is ended first so the simulator exports through its own finally
 * block, and the operator is told that is what will happen before it does.
 */
export function QuitButton() {
  const active = useConsole((s) => s.session.active)
  const pushToast = useConsole((s) => s.pushToast)
  const [asking, setAsking] = useState(false)
  const [busy, setBusy] = useState(false)

  const quit = async () => {
    setBusy(true)
    try {
      await api.quit(active)
      setAsking(false)
      pushToast('The console is stopping. You can close this tab.', 'info')
    } catch (error) {
      if (error instanceof ApiError && error.status === 0) {
        // The backend closed the socket before answering, which is a clean stop.
        setAsking(false)
        pushToast('The console has stopped. You can close this tab.', 'info')
      } else {
        pushToast('The console did not stop', 'bad')
      }
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <Button variant="ghost" onClick={() => setAsking(true)} title="Stop the console and release its port">
        Quit
      </Button>
      <ConfirmModal
        open={asking}
        title="Stop the console?"
        consequence={
          active
            ? 'A run is still going. It will be told to end, which exports its metrics and timeline, and then the console stops and releases its port.'
            : 'The console stops and releases its port. Any browser tab showing it will go quiet.'
        }
        confirmLabel={busy ? 'Stopping' : active ? 'End the run and quit' : 'Quit'}
        onConfirm={quit}
        onCancel={() => setAsking(false)}
      />
    </>
  )
}

/** The workflow is a sequence, so a stage that cannot be entered says why. */
export function ViewSwitcher() {
  const view = useConsole((s) => s.view)
  const setView = useConsole((s) => s.setView)
  const phase = useConsole((s) => s.session.phase)
  const runId = useConsole((s) => s.session.run_id)

  const gate = (id: ViewName): string | null => {
    if (id === 'operations' && phase === 'idle') return 'Launch a run to watch it here'
    if (id === 'setup' && (phase === 'running' || phase === 'paused' || phase === 'preparing'))
      return 'A run is in flight. End it before configuring another'
    if (id === 'author' && (phase === 'running' || phase === 'paused' || phase === 'preparing'))
      return 'A run is in flight. End it before authoring a package'
    if (id === 'debrief' && !runId && phase === 'idle') return 'Finish a run, or open one from the run history'
    return null
  }

  return (
    <nav className="flex overflow-hidden rounded-panel border border-ink-line" aria-label="Workflow stage">
      {VIEWS.map((item) => {
        const blocked = gate(item.id)
        const active = view === item.id
        return (
          <button
            key={item.id}
            type="button"
            onClick={() => !blocked && setView(item.id)}
            disabled={Boolean(blocked)}
            title={blocked ?? undefined}
            aria-current={active ? 'page' : undefined}
            className={`px-4 py-1.5 text-base font-medium transition-colors disabled:cursor-not-allowed disabled:text-ink-faint ${
              active ? 'bg-ink-raised text-ink-text' : 'text-ink-muted hover:bg-ink-raised/60'
            }`}
          >
            {item.label}
          </button>
        )
      })}
    </nav>
  )
}

export function TopBar() {
  const label = useConsole((s) => s.session.label)
  const phase = useConsole((s) => s.session.phase)
  const controllable = phase === 'running' || phase === 'paused'

  return (
    <header className="flex shrink-0 items-center gap-4 border-b border-ink-line bg-ink-panel px-4 py-2.5">
      <div className="flex min-w-0 items-center gap-3">
        <img src="/brand/agent-evac-logo.png" alt="AgentEvac" className="h-8 w-8 rounded" />
        <div className="min-w-0">
          <p className="text-small font-semibold leading-tight">AgentEvac Operator Console</p>
          <p className="truncate text-micro text-ink-faint" title={label}>
            {label || 'no run configured'}
          </p>
        </div>
      </div>

      <div className="mx-auto flex items-center gap-6">
        <RunStatePill />
        <IncidentClock />
        <SpeedSelector disabled={!controllable} />
      </div>

      <div className="flex items-center gap-3">
        <ViewSwitcher />
        <RunControls />
        <ConnectionBadge />
        <QuitButton />
      </div>
    </header>
  )
}

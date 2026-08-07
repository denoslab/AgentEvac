import { type MouseEvent as ReactMouseEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, ApiError } from '../state/api'
import { duration, integer, simClock, titleCase } from '../state/format'
import { useConsole } from '../state/store'
import type {
  PackagesPayload,
  Preview,
  Recording,
  RunConfig,
  ScheduleRow,
  ValidationResult,
} from '../state/types'
import { LiveMap } from '../map/LiveMap'
import { Badge, Button, Collapsible, EmptyState, StatusDot } from '../ui/primitives'

const SPEED_PRESETS = [1, 4, 16, 60, 0]

function PackageCards({
  payload,
  selected,
  onSelect,
}: {
  payload: PackagesPayload
  selected: string
  onSelect: (id: string) => void
}) {
  return (
    <div className="space-y-2">
      {payload.packages.map((pkg) => {
        const active = pkg.id === selected
        return (
          <button
            key={pkg.id}
            type="button"
            disabled={!pkg.usable}
            onClick={() => onSelect(pkg.id)}
            className={`card-choice ${active ? 'card-choice-on' : 'card-choice-off'} ${
              pkg.usable ? '' : 'cursor-not-allowed opacity-50'
            }`}
            title={pkg.problems.join('; ') || pkg.description}
          >
            <div className="flex items-start justify-between gap-2">
              <span className="text-base font-medium">{titleCase(pkg.id)}</span>
              {pkg.has_alert_schedule && (
                <Badge tone="caution" shape="bar">
                  {pkg.alert_waves.length} alert waves
                </Badge>
              )}
            </div>
            <p className="tnum mt-1 text-micro text-ink-muted">
              {integer(pkg.households)} households · {pkg.fire_sources} fire sources ·{' '}
              {pkg.destinations.length} shelters
              {pkg.recommended_horizon_s ? ` · ${duration(pkg.recommended_horizon_s)} horizon` : ''}
            </p>
            {pkg.problems.length > 0 && (
              <p className="mt-1 text-micro text-status-hazard">{pkg.problems[0]}</p>
            )}
            {!pkg.preview_ready && pkg.usable && (
              <p className="mt-1 text-micro text-status-caution">
                map bundle not built, run python -m ui.tools.build_map_assets
              </p>
            )}
          </button>
        )
      })}
    </div>
  )
}

function RegimeSelector({
  options,
  value,
  onChange,
}: {
  options: { id: string; description: string }[]
  value: string
  onChange: (id: string) => void
}) {
  return (
    <div className="space-y-2">
      {options.map((option) => (
        <button
          key={option.id}
          type="button"
          onClick={() => onChange(option.id)}
          className={`card-choice ${value === option.id ? 'card-choice-on' : 'card-choice-off'}`}
        >
          <span className="text-base font-medium">{titleCase(option.id)}</span>
          <p className="mt-0.5 text-micro leading-snug text-ink-muted">{option.description}</p>
        </button>
      ))}
    </div>
  )
}

function EngineSelector({
  options,
  value,
  recordings,
  replayRunId,
  onChange,
  onReplayChange,
  keyPresent,
}: {
  options: { id: string; description: string }[]
  value: string
  recordings: Recording[]
  replayRunId: string | null
  onChange: (id: string) => void
  onReplayChange: (runId: string) => void
  keyPresent: boolean
}) {
  return (
    <div className="space-y-2">
      {options.map((option) => (
        <div key={option.id}>
          <button
            type="button"
            onClick={() => onChange(option.id)}
            className={`card-choice ${value === option.id ? 'card-choice-on' : 'card-choice-off'}`}
          >
            <span className="text-base font-medium">{titleCase(option.id)}</span>
            <p className="mt-0.5 text-micro leading-snug text-ink-muted">{option.description}</p>
            {option.id === 'llm' && !keyPresent && (
              <p className="mt-1 text-micro text-status-caution">
                No OPENAI_API_KEY in this environment, so a live run cannot start.
              </p>
            )}
          </button>
          {option.id === 'replay' && value === 'replay' && (
            <div className="mt-2 pl-3">
              {recordings.length === 0 ? (
                <p className="text-micro text-status-caution">
                  No recorded language-model runs were found under outputs/.
                </p>
              ) : (
                <select
                  className="input"
                  value={replayRunId ?? ''}
                  onChange={(event) => onReplayChange(event.target.value)}
                >
                  <option value="">Choose a recording</option>
                  {recordings.map((recording) => (
                    <option key={recording.run_id} value={recording.run_id}>
                      {recording.run_id} · {recording.package ?? 'unknown package'} ·{' '}
                      {recording.scenario ?? 'unknown regime'} · {recording.size_mb} MB
                    </option>
                  ))}
                </select>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

function IncidentSchedule({ rows, anchor }: { rows: ScheduleRow[]; anchor: string | null }) {
  if (!rows.length) {
    return <p className="p-4 text-small text-ink-faint">This package ships no timed schedule.</p>
  }
  const span = Math.max(1, rows[rows.length - 1].t_s)
  return (
    <div className="p-4">
      <div className="relative h-9 rounded bg-ink-bg">
        {rows.map((row, index) => (
          <span
            key={`${row.kind}-${row.label}-${index}`}
            className={`absolute top-1.5 h-6 w-[3px] rounded ${
              row.kind === 'alert' ? 'bg-status-caution' : 'bg-status-hazard'
            }`}
            style={{ left: `${(row.t_s / span) * 97}%` }}
            title={`${simClock(row.t_s)} ${row.label}`}
          />
        ))}
      </div>
      <div className="tnum mt-1 flex justify-between text-micro text-ink-faint">
        <span>ignition{anchor ? ` · ${anchor}` : ''}</span>
        <span>{duration(span)}</span>
      </div>
      <ul className="mt-3 max-h-40 space-y-1 overflow-auto pr-1">
        {rows.map((row, index) => (
          <li key={`${row.kind}-${row.label}-${index}`} className="flex items-baseline gap-2 text-micro">
            <StatusDot shape={row.kind === 'alert' ? 'bar' : 'flame'} tone={row.kind === 'alert' ? 'caution' : 'hazard'} />
            <span className="tnum w-16 shrink-0 text-ink-muted">{simClock(row.t_s)}</span>
            <span className="text-ink-text">{row.label.replace(/_/g, ' ')}</span>
            <span className="truncate text-ink-faint">{row.detail.replace(/_/g, ' ')}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function ValidationSummary({ result }: { result: ValidationResult | null }) {
  if (!result || (result.problems.length === 0 && result.warnings.length === 0)) return null
  return (
    <div className="space-y-1.5">
      {result.problems.map((issue, index) => (
        <div key={`p${index}`} className="rounded-panel border border-status-hazard/60 bg-status-hazard/10 px-3 py-2">
          <p className="text-small font-medium text-status-hazard">{issue.message}</p>
          <p className="text-micro text-ink-muted">{issue.hint}</p>
        </div>
      ))}
      {result.warnings.map((issue, index) => (
        <div key={`w${index}`} className="rounded-panel border border-status-caution/50 bg-status-caution/10 px-3 py-2">
          <p className="text-small font-medium text-status-caution">{issue.message}</p>
          <p className="text-micro text-ink-muted">{issue.hint}</p>
        </div>
      ))}
    </div>
  )
}

/** The settings sections, in the order an operator works through them. */
const SECTIONS = ['package', 'regime', 'timing', 'engine', 'identity', 'advanced'] as const
type SectionName = (typeof SECTIONS)[number]

//: Primary decisions start open. The rest fold away behind their summaries.
const SECTIONS_INITIALLY_OPEN: Record<SectionName, boolean> = {
  package: true,
  regime: true,
  timing: true,
  engine: false,
  identity: false,
  advanced: false,
}

export function SetupView() {
  const pushToast = useConsole((s) => s.pushToast)
  const setPreview = (preview: Preview | null) =>
    useConsole.setState({ preview: preview ? { ...preview, type: 'preview' } : null })

  const [payload, setPayload] = useState<PackagesPayload | null>(null)
  const [recordings, setRecordings] = useState<Recording[]>([])
  const [schedule, setSchedule] = useState<ScheduleRow[]>([])
  const [config, setConfig] = useState<Partial<RunConfig>>({})
  const [validation, setValidation] = useState<ValidationResult | null>(null)
  const [launching, setLaunching] = useState(false)
  const [keyPresent, setKeyPresent] = useState(true)
  const [environmentProblem, setEnvironmentProblem] = useState<string | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [openSections, setOpenSections] = useState<Record<SectionName, boolean>>(SECTIONS_INITIALLY_OPEN)
  const [scheduleOpen, setScheduleOpen] = useState(true)
  const [scheduleHeight, setScheduleHeight] = useState(280)
  const previewToken = useRef(0)

  const section = (name: SectionName) => ({
    open: openSections[name],
    onOpenChange: (open: boolean) => setOpenSections((previous) => ({ ...previous, [name]: open })),
  })
  const setAllSections = (open: boolean) =>
    setOpenSections(Object.fromEntries(SECTIONS.map((name) => [name, open])) as Record<SectionName, boolean>)
  const anyClosed = SECTIONS.some((name) => !openSections[name])

  /** Drag the divider above the schedule to trade its height against the map. */
  const endDrag = useRef<(() => void) | null>(null)
  useEffect(() => () => endDrag.current?.(), [])

  const startScheduleResize = (event: ReactMouseEvent) => {
    event.preventDefault()
    const startY = event.clientY
    const startHeight = scheduleHeight
    const onMove = (moveEvent: MouseEvent) => {
      const next = startHeight - (moveEvent.clientY - startY)
      setScheduleHeight(Math.min(560, Math.max(120, next)))
    }
    const onUp = () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      endDrag.current = null
    }
    // Leaving the view mid-drag must not strand the cursor or the listeners.
    endDrag.current = onUp
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    document.body.style.cursor = 'row-resize'
    document.body.style.userSelect = 'none'
  }

  const nudgeScheduleHeight = (delta: number) =>
    setScheduleHeight((current) => Math.min(560, Math.max(120, current + delta)))

  // ------------------------------------------------------------------- load
  useEffect(() => {
    let cancelled = false
    Promise.all([api.packages(), api.recordings(), api.health()])
      .then(([packagesPayload, recordingsPayload, health]) => {
        if (cancelled) return
        setPayload(packagesPayload)
        setRecordings(recordingsPayload.recordings)
        setKeyPresent(health.environment.openai_key)
        // Everything a run needs is checked before the operator configures one,
        // so a missing piece of the environment is not discovered on launch.
        setEnvironmentProblem(
          !health.environment.simulator_python
            ? (health.environment.simulator_python_hint ??
              'No interpreter on this machine can start a simulation.')
            : !health.environment.sumo_home_exists
              ? `SUMO_HOME points at ${health.environment.sumo_home}, which does not exist.`
              : !health.environment.sumo_binary
                ? 'The sumo binary is not on the path, so a run cannot start.'
                : null,
        )
        const previous = useConsole.getState().session.config
        const first = packagesPayload.packages.find((p) => p.usable)
        setConfig({
          ...packagesPayload.defaults,
          ...previous,
          package:
            (previous.package as string) ??
            (packagesPayload.packages.some((p) => p.id === packagesPayload.defaults.package && p.usable)
              ? packagesPayload.defaults.package
              : (first?.id ?? packagesPayload.defaults.package)),
        })
      })
      .catch((error: unknown) => {
        if (!cancelled) setLoadError(error instanceof ApiError ? error.message : String(error))
      })
    return () => {
      cancelled = true
    }
  }, [])

  const selectedPackage = useMemo(
    () => payload?.packages.find((p) => p.id === config.package) ?? null,
    [payload, config.package],
  )

  // ------------------------------------------------------- package follow-ups
  useEffect(() => {
    if (!config.package) return
    const token = ++previewToken.current
    setPreview(null)
    setSchedule([])
    api
      .packageSchedule(config.package)
      .then((data) => {
        if (token === previewToken.current) setSchedule(data.schedule)
      })
      .catch(() => undefined)
    api
      .packagePreview(config.package)
      .then((preview) => {
        if (token === previewToken.current) setPreview(preview)
      })
      .catch((error: unknown) => {
        if (token !== previewToken.current) return
        const hint =
          error instanceof ApiError && (error.payload as { hint?: string })?.hint
            ? (error.payload as { hint: string }).hint
            : 'The map bundle for this package has not been built.'
        pushToast(hint, 'warn')
      })
  }, [config.package, pushToast])

  // Follow the package's own recommended horizon unless the operator overrode it.
  const horizonTouched = useRef(false)
  useEffect(() => {
    if (horizonTouched.current || !selectedPackage?.recommended_horizon_s) return
    setConfig((previous) => ({ ...previous, sim_end_time_s: selectedPackage.recommended_horizon_s! }))
  }, [selectedPackage])

  // ------------------------------------------------------------- validation
  useEffect(() => {
    if (!config.package) return
    const handle = window.setTimeout(() => {
      api
        .validate(config)
        .then(setValidation)
        .catch(() => undefined)
    }, 220)
    return () => window.clearTimeout(handle)
  }, [config])

  const update = useCallback(
    (patch: Partial<RunConfig>) => setConfig((previous) => ({ ...previous, ...patch })),
    [],
  )

  const launch = async () => {
    setLaunching(true)
    try {
      const result = await api.launch(config)
      useConsole.getState().setSession(result.session)
      useConsole.getState().setView('operations')
      pushToast('Run launched. The road network loads before the first step.', 'good')
    } catch (error) {
      if (error instanceof ApiError && error.status === 400) {
        setValidation(error.payload as ValidationResult)
        pushToast('The run configuration has a problem that must be fixed first.', 'warn')
      } else {
        pushToast(error instanceof ApiError ? error.message : 'launch failed', 'bad')
      }
    } finally {
      setLaunching(false)
    }
  }

  if (loadError) {
    return (
      <EmptyState
        title="The console backend is not reachable"
        detail={loadError}
        action={
          <Button onClick={() => window.location.reload()} variant="primary">
            Try again
          </Button>
        }
      />
    )
  }

  if (!payload) {
    return <EmptyState title="Loading scenario packages" />
  }

  const blocked = validation ? !validation.ok : false
  const blockingReason = validation?.problems[0]?.message

  return (
    <div className="flex h-full min-h-0 gap-3 p-3">
      {/* Settings in priority order. Every section folds, and a folded one keeps
          its current value in the heading, so the column scrolls instead of
          growing past the window. */}
      <div className="flex w-[420px] shrink-0 flex-col gap-3 overflow-y-auto pr-1">
        <div className="flex shrink-0 items-center justify-between px-1">
          <h2 className="text-small font-semibold text-ink-muted">Run configuration</h2>
          <button
            type="button"
            onClick={() => setAllSections(anyClosed)}
            className="text-micro text-ink-faint underline underline-offset-2 hover:text-ink-text"
          >
            {anyClosed ? 'Expand all' : 'Collapse all'}
          </button>
        </div>

        <Collapsible
          title="Scenario package"
          {...section('package')}
          meta={selectedPackage ? `${titleCase(selectedPackage.id)} · ${integer(selectedPackage.households)} households` : 'none chosen'}
        >
          <PackageCards payload={payload} selected={config.package ?? ''} onSelect={(id) => update({ package: id })} />
        </Collapsible>

        <Collapsible
          title="Warning info"
          {...section('regime')}
          meta={titleCase(String(config.scenario ?? ''))}
        >
          <RegimeSelector
            options={payload.scenarios}
            value={config.scenario ?? ''}
            onChange={(id) => update({ scenario: id })}
          />
        </Collapsible>

        <Collapsible
          title="Alert timing"
          {...section('timing')}
          meta={
            !selectedPackage?.has_alert_schedule
              ? 'no schedule'
              : Number(config.alert_minutes_earlier ?? 0) === 0
                ? 'historical timing'
                : `${Math.round(Number(config.alert_minutes_earlier))} min earlier`
          }
        >
          <label className="field-label" htmlFor="alert-offset">
            Orders issued{' '}
            <span className="tnum font-semibold text-status-caution">
              {Math.round(Number(config.alert_minutes_earlier ?? 0))} minutes
            </span>{' '}
            earlier than the record
          </label>
          <input
            id="alert-offset"
            type="range"
            min={0}
            max={60}
            step={5}
            disabled={!selectedPackage?.has_alert_schedule}
            value={Number(config.alert_minutes_earlier ?? 0)}
            onChange={(event) => update({ alert_minutes_earlier: Number(event.target.value) })}
            className="w-full accent-[#E69F00] disabled:opacity-40"
          />
          <div className="tnum mt-1 flex justify-between text-micro text-ink-faint">
            <span>historical</span>
            <span>60 min earlier</span>
          </div>
          {!selectedPackage?.has_alert_schedule && (
            <p className="mt-2 text-micro text-ink-faint">
              This package ships no alert schedule, so there is no order timing to shift.
            </p>
          )}
          {selectedPackage?.has_alert_schedule && (
            <ul className="tnum mt-3 space-y-0.5 text-micro text-ink-muted">
              {selectedPackage.alert_waves.map((wave) => {
                const shifted = wave.issue_time_s - Number(config.alert_minutes_earlier ?? 0) * 60
                return (
                  <li key={wave.id} className="flex justify-between gap-2">
                    <span>{wave.id}</span>
                    <span>
                      {simClock(wave.issue_time_s)}
                      {shifted !== wave.issue_time_s && (
                        <span className="text-status-caution"> → {simClock(Math.max(0, shifted))}</span>
                      )}
                    </span>
                  </li>
                )
              })}
            </ul>
          )}
        </Collapsible>

        <Collapsible
          title="Agent type"
          {...section('engine')}
          meta={
            config.engine === 'replay'
              ? `Replay · ${config.replay_run_id ?? 'no recording'}`
              : titleCase(String(config.engine ?? ''))
          }
        >
          <EngineSelector
            options={payload.engines}
            value={config.engine ?? ''}
            recordings={recordings}
            replayRunId={(config.replay_run_id as string) ?? null}
            onChange={(id) => update({ engine: id })}
            onReplayChange={(runId) => update({ replay_run_id: runId })}
            keyPresent={keyPresent}
          />
        </Collapsible>

        <Collapsible
          title="Run identity"
          {...section('identity')}
          meta={`seed ${config.seed ?? '--'}${config.label ? ` · ${config.label}` : ''}`}
        >
          <label className="field-label" htmlFor="run-label">
            Label
          </label>
          <input
            id="run-label"
            className="input"
            placeholder="Tantallon reconstruction, historical alert timing"
            value={config.label ?? ''}
            onChange={(event) => update({ label: event.target.value })}
          />
          <label className="field-label mt-3" htmlFor="run-seed">
            Seed
          </label>
          <div className="flex gap-2">
            <input
              id="run-seed"
              className="input tnum"
              inputMode="numeric"
              value={String(config.seed ?? '')}
              onChange={(event) => update({ seed: Number(event.target.value.replace(/\D/g, '')) || 0 })}
            />
            <Button onClick={() => update({ seed: Math.floor(Math.random() * 100000) })}>Randomize</Button>
          </div>
        </Collapsible>

        <Collapsible
          title="Advanced"
          {...section('advanced')}
          meta={`${duration(Number(config.sim_end_time_s ?? 0))} · ${config.messaging ? 'messaging on' : 'messaging off'}`}
        >
          <label className="field-label" htmlFor="horizon">
            Horizon, seconds after ignition
          </label>
          <input
            id="horizon"
            className="input tnum"
            inputMode="numeric"
            value={String(config.sim_end_time_s ?? '')}
            onChange={(event) => {
              horizonTouched.current = true
              update({ sim_end_time_s: Number(event.target.value.replace(/\D/g, '')) || 0 })
            }}
          />
          <p className="mt-1 text-micro text-ink-faint">{duration(Number(config.sim_end_time_s ?? 0))} of incident time</p>

          <label className="field-label mt-3" htmlFor="cadence">
            Decision round every, seconds
          </label>
          <input
            id="cadence"
            className="input tnum"
            inputMode="numeric"
            value={String(config.decision_period_s ?? '')}
            onChange={(event) => update({ decision_period_s: Number(event.target.value.replace(/\D/g, '')) || 0 })}
          />

          <label className="mt-3 flex items-center gap-2 text-base">
            <input
              type="checkbox"
              className="h-4 w-4 accent-[#009E73]"
              checked={Boolean(config.messaging)}
              onChange={(event) => update({ messaging: event.target.checked })}
            />
            Households exchange messages with their neighbours
          </label>

          <label className="field-label mt-3" htmlFor="initial-speed">
            Starting speed
          </label>
          <select
            id="initial-speed"
            className="input"
            value={String(config.initial_speed ?? 16)}
            onChange={(event) => update({ initial_speed: Number(event.target.value) })}
          >
            {SPEED_PRESETS.map((speed) => (
              <option key={speed} value={speed}>
                {speed === 0 ? 'as fast as the machine allows' : `${speed}x real time`}
              </option>
            ))}
          </select>
        </Collapsible>

        <div className="sticky bottom-0 shrink-0 space-y-2 bg-ink-bg pb-1 pt-2">
          {environmentProblem && (
            <div className="rounded-panel border border-status-hazard/60 bg-status-hazard/10 px-3 py-2">
              <p className="text-small font-medium text-status-hazard">
                This machine cannot start a run
              </p>
              <p className="text-micro text-ink-muted">{environmentProblem}</p>
            </div>
          )}
          <ValidationSummary result={validation} />
          <Button
            onClick={launch}
            variant="primary"
            busy={launching}
            disabled={blocked || Boolean(environmentProblem)}
            disabledReason={environmentProblem ?? blockingReason}
            className="h-11 w-full text-panel"
          >
            Launch run
          </Button>
        </div>
      </div>

      {/* What the operator is about to run, before they run it. The schedule
          folds away and its divider drags, so the map can take the room. */}
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="min-h-0 flex-1">
          <LiveMap />
        </div>

        {scheduleOpen && (
          <div
            role="separator"
            aria-orientation="horizontal"
            aria-label="Resize the incident schedule"
            aria-valuenow={scheduleHeight}
            aria-valuemin={120}
            aria-valuemax={560}
            tabIndex={0}
            onMouseDown={startScheduleResize}
            onKeyDown={(event) => {
              if (event.key === 'ArrowUp') nudgeScheduleHeight(24)
              else if (event.key === 'ArrowDown') nudgeScheduleHeight(-24)
              else return
              event.preventDefault()
            }}
            className="group flex h-3 shrink-0 cursor-row-resize items-center justify-center"
          >
            <span className="h-[3px] w-10 rounded-full bg-ink-line transition-colors group-hover:bg-status-caution" />
          </div>
        )}

        <section
          className="panel flex shrink-0 flex-col"
          style={scheduleOpen ? { height: scheduleHeight } : undefined}
        >
          <header className="flex shrink-0 items-center justify-between border-b border-ink-line">
            <button
              type="button"
              onClick={() => setScheduleOpen((open) => !open)}
              aria-expanded={scheduleOpen}
              className="flex flex-1 items-center gap-2 px-4 py-2.5 text-left text-small font-semibold uppercase tracking-wide text-ink-muted hover:text-ink-text"
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 12 12"
                aria-hidden
                className={`transition-transform ${scheduleOpen ? 'rotate-90' : ''}`}
              >
                <path d="M4 2 L8 6 L4 10" fill="none" stroke="currentColor" strokeWidth="1.6" />
              </svg>
              Incident schedule
              <span className="tnum ml-1 font-normal normal-case text-ink-faint">
                {schedule.length ? `${schedule.length} events` : 'none'}
              </span>
            </button>
          </header>
          {scheduleOpen && (
            <div className="min-h-0 flex-1 overflow-auto">
              <IncidentSchedule rows={schedule} anchor={selectedPackage?.anchor_clock ?? null} />
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

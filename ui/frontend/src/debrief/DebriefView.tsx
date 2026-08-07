import uPlot from 'uplot'
import { useEffect, useMemo, useRef, useState } from 'react'
import { api, ApiError } from '../state/api'
import { decimal, duration, integer, percent, simClock, titleCase, wallDate } from '../state/format'
import { useConsole } from '../state/store'
import type { MetricsPayload, RunRecord } from '../state/types'
import { Badge, Button, Collapsible, EmptyState, Panel, Spinner, StatTile } from '../ui/primitives'

/**
 * Several metrics are exported as a dictionary holding the headline number
 * beside its per-household breakdown. This reads the headline wherever it lives,
 * so a plain number and a summary object both render.
 */
function scalar(value: unknown, ...keys: string[]): number | null {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null
  if (value && typeof value === 'object') {
    for (const key of keys) {
      const candidate = (value as Record<string, unknown>)[key]
      if (typeof candidate === 'number' && Number.isFinite(candidate)) return candidate
    }
  }
  return null
}

function OutcomeTiles({ payload }: { payload: MetricsPayload }) {
  const m = payload.metrics
  const total = Number(m.total_agents ?? 0)
  const arrived = Number(m.arrived_agents ?? 0)
  const contact = Number(m.fire_contact?.agents_ever_in_contact ?? 0)
  const strandedByFire = Number(m.non_evacuated_reached_by_fire?.count ?? 0)
  const mobilization = scalar(m.mobilization_delay, 'average')
  const travel = scalar(m.average_travel_time, 'average')
  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      <StatTile
        label="Households cleared"
        value={`${integer(arrived)} / ${integer(total)}`}
        hint={total ? percent(arrived / total) : undefined}
        tone={total && arrived === total ? 'nominal' : 'caution'}
        large
        shape="square"
      />
      <StatTile
        label="Mean mobilization delay"
        value={mobilization == null ? '--' : duration(mobilization)}
        hint="from the order reaching a household to it leaving"
        tone="neutral"
        large
      />
      <StatTile
        label="Reached by fire"
        value={integer(contact + strandedByFire)}
        hint={`${integer(contact)} while evacuating, ${integer(strandedByFire)} still at home`}
        tone={contact + strandedByFire > 0 ? 'hazard' : 'nominal'}
        large
        shape="flame"
      />
      <StatTile
        label="Mean travel time"
        value={travel == null ? '--' : duration(travel)}
        hint="door to shelter, over households that arrived"
        tone="neutral"
        large
        shape="triangle"
      />
    </div>
  )
}

function ClearanceTable({ payload }: { payload: MetricsPayload }) {
  const rows = Object.entries((payload.metrics.area_clearance ?? {}) as Record<string, any>)
  if (!rows.length) return <p className="text-small text-ink-faint">This run defined no alert areas.</p>
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-small">
        <thead>
          <tr className="border-b border-ink-line text-micro uppercase tracking-wide text-ink-muted">
            <th className="py-2 text-left font-semibold">Community</th>
            <th className="py-2 text-right font-semibold">Ordered</th>
            <th className="py-2 text-right font-semibold">Order time</th>
            <th className="py-2 text-right font-semibold">Cleared by</th>
            <th className="py-2 text-right font-semibold">Time to clear</th>
            <th className="py-2 text-right font-semibold">Channel</th>
          </tr>
        </thead>
        <tbody className="tnum">
          {rows
            .sort((a, b) => (a[1].order_t_s ?? 0) - (b[1].order_t_s ?? 0))
            .map(([name, info]) => {
              const span =
                info.clearance_t_s != null && info.order_t_s != null
                  ? Number(info.clearance_t_s) - Number(info.order_t_s)
                  : null
              return (
                <tr key={name} className="border-b border-ink-line/50 last:border-0">
                  <td className="py-2 text-left">{name.replace(/_/g, ' ')}</td>
                  <td className="py-2 text-right">
                    {integer(info.departed)} / {integer(info.ordered)}
                  </td>
                  <td className="py-2 text-right">{simClock(info.order_t_s)}</td>
                  <td className="py-2 text-right">
                    {info.fully_cleared ? simClock(info.clearance_t_s) : <span className="text-status-caution">not cleared</span>}
                  </td>
                  <td className="py-2 text-right">{span == null ? '--' : duration(span)}</td>
                  <td className="py-2 text-right text-ink-muted">{info.channel}</td>
                </tr>
              )
            })}
        </tbody>
      </table>
    </div>
  )
}

function ShareRows({ counts, fractions }: { counts: Record<string, number>; fractions?: Record<string, number> }) {
  const entries = Object.entries(counts ?? {}).sort((a, b) => b[1] - a[1])
  if (!entries.length) return <p className="text-small text-ink-faint">No data recorded.</p>
  const total = entries.reduce((sum, [, value]) => sum + value, 0)
  return (
    <ul className="space-y-2">
      {entries.map(([name, value]) => {
        const fraction = fractions?.[name] ?? (total ? value / total : 0)
        return (
          <li key={name}>
            <div className="flex justify-between text-small">
              <span>{name.replace(/_/g, ' ')}</span>
              <span className="tnum text-ink-muted">
                {integer(value)} · {percent(fraction)}
              </span>
            </div>
            <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-ink-raised">
              <div className="h-full rounded-full bg-status-moving" style={{ width: `${fraction * 100}%` }} />
            </div>
          </li>
        )
      })}
    </ul>
  )
}

function ComparisonCurve({
  primary,
  comparison,
}: {
  primary: MetricsPayload
  comparison: MetricsPayload | null
}) {
  const container = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(600)

  useEffect(() => {
    if (!container.current) return
    const observer = new ResizeObserver((entries) => setWidth(Math.max(320, Math.floor(entries[0].contentRect.width))))
    observer.observe(container.current)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (!container.current) return
    // uPlot needs one shared x axis, so both runs are resampled onto the union
    // of their event times.
    const times = new Set<number>([0])
    for (const point of primary.curve.departures) times.add(point[0])
    for (const point of primary.curve.arrivals) times.add(point[0])
    if (comparison) {
      for (const point of comparison.curve.departures) times.add(point[0])
      for (const point of comparison.curve.arrivals) times.add(point[0])
    }
    const xs = [...times].sort((a, b) => a - b)
    const stepAt = (series: [number, number][], t: number) => {
      let value = 0
      for (const [time, count] of series) {
        if (time > t) break
        value = count
      }
      return value
    }
    const series: uPlot.Series[] = [
      {},
      { label: 'left home', stroke: '#0072B2', width: 2, points: { show: false } },
      { label: 'arrived', stroke: '#009E73', width: 2, points: { show: false } },
    ]
    const data: uPlot.AlignedData = [
      xs,
      xs.map((t) => stepAt(primary.curve.departures, t)),
      xs.map((t) => stepAt(primary.curve.arrivals, t)),
    ] as uPlot.AlignedData
    if (comparison) {
      series.push({ label: 'left home, comparison', stroke: '#0072B2', width: 1.5, dash: [4, 3], points: { show: false } })
      series.push({ label: 'arrived, comparison', stroke: '#009E73', width: 1.5, dash: [4, 3], points: { show: false } })
      ;(data as number[][]).push(xs.map((t) => stepAt(comparison.curve.departures, t)))
      ;(data as number[][]).push(xs.map((t) => stepAt(comparison.curve.arrivals, t)))
    }

    const instance = new uPlot(
      {
        width,
        height: 260,
        padding: [10, 12, 0, 0],
        legend: { show: false },
        cursor: { drag: { x: true, y: false } },
        scales: { x: { time: false } },
        axes: [
          {
            stroke: '#6C7681',
            grid: { stroke: '#232B35' },
            ticks: { stroke: '#232B35' },
            font: '11px Inter, system-ui, sans-serif',
            values: (_self, splits) => splits.map((value) => simClock(value)),
          },
          {
            stroke: '#6C7681',
            grid: { stroke: '#232B35' },
            ticks: { stroke: '#232B35' },
            font: '11px Inter, system-ui, sans-serif',
            size: 42,
          },
        ],
        series,
      },
      data,
      container.current,
    )
    return () => instance.destroy()
  }, [primary, comparison, width])

  return <div ref={container} className="w-full" />
}

function MetadataFooter({ payload }: { payload: MetricsPayload }) {
  const record = payload.record
  const artifacts = payload.artifacts
  return (
    <Panel title="Run configuration and files">
      <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-small lg:grid-cols-4">
        {(
          [
            ['Run', record.run_id],
            ['Package', record.package],
            ['Information regime', record.scenario],
            ['Decision engine', record.agent_type],
            ['Seed', record.seed],
            ['Horizon', record.horizon_s ? duration(record.horizon_s) : null],
            [
              'Alert offset',
              record.alert_offset_s == null
                ? null
                : record.alert_offset_s === 0
                  ? 'historical timing'
                  : `${Math.round(Math.abs(record.alert_offset_s) / 60)} min ${record.alert_offset_s < 0 ? 'earlier' : 'later'}`,
            ],
            ['Finished', wallDate(record.modified)],
          ] as [string, unknown][]
        ).map(([label, value]) => (
          <div key={label}>
            <dt className="text-micro uppercase tracking-wide text-ink-faint">{label}</dt>
            <dd className="tnum truncate font-mono text-small">{value == null ? '--' : String(value)}</dd>
          </div>
        ))}
      </dl>
      <div className="mt-4 flex flex-wrap gap-2">
        <a className="btn btn-primary px-3" href={api.exportUrl(payload.run_id)} download>
          Download everything
        </a>
        {Object.entries(artifacts)
          .filter(([, path]) => Boolean(path))
          .map(([name, path]) => (
            <a key={name} className="btn px-3" href={api.fileUrl(String(path))} download title={String(path)}>
              {titleCase(name)}
            </a>
          ))}
      </div>
    </Panel>
  )
}

export function DebriefView() {
  const session = useConsole((s) => s.session)
  const setView = useConsole((s) => s.setView)
  const pushToast = useConsole((s) => s.pushToast)

  const [runs, setRuns] = useState<RunRecord[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [payload, setPayload] = useState<MetricsPayload | null>(null)
  const [comparisonId, setComparisonId] = useState<string>('')
  const [comparison, setComparison] = useState<MetricsPayload | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const consoleRunId = (session.artifacts?.sim_run_id as string) ?? null

  useEffect(() => {
    api
      .history()
      .then((data) => {
        setRuns(data.runs)
        setSelected((current) => current ?? consoleRunId ?? data.runs[0]?.run_id ?? null)
      })
      .catch((err: unknown) => setError(err instanceof ApiError ? err.message : String(err)))
  }, [consoleRunId, session.phase])

  useEffect(() => {
    if (!selected) return
    setLoading(true)
    setError(null)
    api
      .runMetrics(selected)
      .then(setPayload)
      .catch((err: unknown) => setError(err instanceof ApiError ? err.message : String(err)))
      .finally(() => setLoading(false))
  }, [selected])

  useEffect(() => {
    if (!comparisonId) {
      setComparison(null)
      return
    }
    api
      .runMetrics(comparisonId)
      .then(setComparison)
      .catch(() => {
        setComparison(null)
        pushToast('That run could not be loaded for comparison.', 'warn')
      })
  }, [comparisonId, pushToast])

  const mismatches = useMemo(() => {
    if (!payload || !comparison) return []
    const out: string[] = []
    const fields: [keyof RunRecord, string][] = [
      ['package', 'scenario package'],
      ['scenario', 'information regime'],
      ['agent_type', 'decision engine'],
      ['horizon_s', 'horizon'],
      ['seed', 'seed'],
    ]
    for (const [key, label] of fields) {
      if (payload.record[key] !== comparison.record[key]) {
        out.push(`${label}: ${String(payload.record[key])} against ${String(comparison.record[key])}`)
      }
    }
    return out
  }, [payload, comparison])

  const partial = session.phase === 'ended_by_operator' || session.phase === 'failed'

  if (error && !payload) {
    return <EmptyState title="No finished run could be read" detail={error} />
  }
  if (!selected || (loading && !payload)) {
    return (
      <EmptyState title={loading ? 'Loading the run' : 'No finished runs yet'} detail={loading ? undefined : 'Launch a run from Setup, and its results appear here when it finishes.'} />
    )
  }
  if (!payload) return <EmptyState title="Loading the run" />

  const m = payload.metrics

  return (
    <div className="h-full min-h-0 overflow-y-auto">
      <div className="mx-auto max-w-[1400px] space-y-3 p-3">
        <header className="flex flex-wrap items-center gap-3">
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-view font-semibold">{payload.record.label}</h1>
            <p className="text-small text-ink-muted">
              {titleCase(String(payload.record.package ?? 'unknown package'))} ·{' '}
              {titleCase(String(payload.record.scenario ?? ''))} · {wallDate(payload.record.modified)}
            </p>
          </div>
          <label className="flex items-center gap-2 text-small">
            <span className="text-ink-muted">Run</span>
            <select className="input w-72" value={selected} onChange={(event) => setSelected(event.target.value)}>
              {runs.map((run) => (
                <option key={run.run_id} value={run.run_id}>
                  {run.from_console ? '★ ' : ''}
                  {run.label}
                </option>
              ))}
            </select>
          </label>
          <Button variant="primary" onClick={() => setView('setup')}>
            Start a new run
          </Button>
        </header>

        {partial && (
          <div className="rounded-panel border border-status-caution/50 bg-status-caution/10 px-3 py-2 text-small text-status-caution">
            This run did not reach its horizon, so the results below cover only the part that ran.
          </div>
        )}

        <OutcomeTiles payload={payload} />

        <Panel
          title="Evacuation curve"
          action={
            <label className="flex items-center gap-2 text-micro font-normal normal-case">
              <span className="text-ink-faint">compare with</span>
              <select
                className="rounded border border-ink-line bg-ink-bg px-2 py-1 text-micro"
                value={comparisonId}
                onChange={(event) => setComparisonId(event.target.value)}
              >
                <option value="">no comparison</option>
                {runs
                  .filter((run) => run.run_id !== selected)
                  .map((run) => (
                    <option key={run.run_id} value={run.run_id}>
                      {run.label}
                    </option>
                  ))}
              </select>
            </label>
          }
        >
          <ComparisonCurve primary={payload} comparison={comparison} />
          {comparison && mismatches.length > 0 && (
            <p className="mt-2 text-micro text-status-caution">
              These runs differ in {mismatches.join('; ')}. Read the comparison with that in mind.
            </p>
          )}
          {comparison && mismatches.length === 0 && (
            <p className="mt-2 text-micro text-ink-faint">
              Both runs share their package, regime, engine, horizon, and seed.
            </p>
          )}
        </Panel>

        <Panel title="Clearance by community">
          <ClearanceTable payload={payload} />
        </Panel>

        <div className="grid gap-3 lg:grid-cols-2">
          <Collapsible title="Order compliance" defaultOpen meta={percent(m.compliance?.overall?.rate)}>
            <p className="mb-3 text-small text-ink-muted">
              Households under an order that had left home by the end of the run.
            </p>
            <ShareRows
              counts={Object.fromEntries(
                Object.entries((m.compliance?.by_channel ?? {}) as Record<string, any>).map(([k, v]) => [
                  k,
                  v.evacuated,
                ]),
              )}
              fractions={Object.fromEntries(
                Object.entries((m.compliance?.by_channel ?? {}) as Record<string, any>).map(([k, v]) => [k, v.rate]),
              )}
            />
          </Collapsible>

          <Collapsible title="How households first learned" defaultOpen meta={`${integer(m.n_aware)} warned`}>
            <ShareRows counts={m.awareness_source_share?.counts ?? {}} fractions={m.awareness_source_share?.share} />
            {Number(m.n_never_aware ?? 0) > 0 && (
              <p className="mt-3 text-small text-status-caution">
                {integer(m.n_never_aware)} households were never warned by any channel.
              </p>
            )}
          </Collapsible>

          <Collapsible title="Where households went">
            <ShareRows
              counts={m.destination_choice_share?.counts ?? {}}
              fractions={m.destination_choice_share?.fractions}
            />
          </Collapsible>

          <Collapsible title="Why households left">
            <ShareRows counts={m.departure_reasons ?? {}} />
          </Collapsible>

          <Collapsible title="Decision quality">
            <dl className="space-y-2 text-small">
              {(
                [
                  [
                    'Route choice entropy',
                    decimal(scalar(m.route_choice_entropy), 3),
                    'spread of destination choices, in nats',
                  ],
                  [
                    'Decision changes per household',
                    decimal(scalar(m.decision_instability, 'average_changes'), 2),
                    'how often a household reversed a choice it had already made',
                  ],
                  [
                    'Departure time spread',
                    // The simulator exports the variance, and seconds read better
                    // than seconds squared.
                    duration(Math.sqrt(Math.max(0, scalar(m.departure_time_variability) ?? 0))),
                    'standard deviation of departure times',
                  ],
                  [
                    'Mean hazard exposure',
                    decimal(scalar(m.average_hazard_exposure, 'global_average'), 4),
                    'exposure index over evacuating households',
                  ],
                  [
                    'Mean signal conflict',
                    decimal(scalar(m.average_signal_conflict, 'global_average'), 4),
                    'disagreement between what households saw and what they were told',
                  ],
                ] as [string, string, string][]
              ).map(([label, value, hint]) => (
                <div key={label} className="flex items-baseline justify-between gap-4 border-b border-ink-line/50 pb-1.5 last:border-0">
                  <div>
                    <dt className="text-ink-text">{label}</dt>
                    <dd className="text-micro text-ink-faint">{hint}</dd>
                  </div>
                  <span className="tnum text-panel font-semibold">{value}</span>
                </div>
              ))}
            </dl>
          </Collapsible>

          {m.time_margin?.margin_s && (
            <Collapsible
              title="Headroom before the fire"
              meta={`${integer(m.time_margin.n_threatened)} households threatened`}
            >
              <p className="mb-3 text-small text-ink-muted">
                Time between a household leaving and a fire front reaching its street. A negative
                margin means the fire arrived first.
              </p>
              <dl className="space-y-1.5 text-small">
                {(
                  [
                    ['Median', m.time_margin.margin_s.median],
                    ['Mean', m.time_margin.margin_s.mean],
                    ['Worst tenth', m.time_margin.margin_s.p10],
                    ['Smallest', m.time_margin.margin_s.min],
                  ] as [string, number][]
                ).map(([label, value]) => (
                  <div key={label} className="flex justify-between border-b border-ink-line/50 pb-1 last:border-0">
                    <dt className="text-ink-muted">{label}</dt>
                    <dd className={`tnum ${value < 0 ? 'text-status-hazard' : ''}`}>{duration(value)}</dd>
                  </div>
                ))}
              </dl>
              {Number(m.time_margin.n_caught_at_home ?? 0) > 0 && (
                <p className="mt-3 text-small text-status-hazard">
                  {integer(m.time_margin.n_caught_at_home)} households were still at home when a front
                  reached their street.
                </p>
              )}
              {Number(m.time_margin.n_censored ?? 0) > 0 && (
                <p className="mt-1 text-micro text-ink-faint">
                  {integer(m.time_margin.n_censored)} households were never reached by a front, so their
                  margin is unbounded and excluded here.
                </p>
              )}
            </Collapsible>
          )}

          {m.token_usage && (
            <Collapsible title="Language-model usage" meta={`${integer(m.token_usage.llm_calls)} calls`}>
              <dl className="space-y-1.5 text-small">
                {Object.entries(m.token_usage as Record<string, number>).map(([key, value]) => (
                  <div key={key} className="flex justify-between border-b border-ink-line/50 pb-1 last:border-0">
                    <dt className="text-ink-muted">{titleCase(key)}</dt>
                    <dd className="tnum">{integer(value)}</dd>
                  </div>
                ))}
              </dl>
            </Collapsible>
          )}
        </div>

        {Number(m.non_evacuated_reached_by_fire?.count ?? 0) > 0 && (
          <Panel title="Households the fire reached at home">
            <p className="text-small text-ink-muted">
              {integer(m.non_evacuated_reached_by_fire.count)} households had a fire front reach their street before
              they left.
            </p>
            <div className="mt-2 flex flex-wrap gap-1">
              {(m.non_evacuated_reached_by_fire.agent_ids as string[]).slice(0, 60).map((id) => (
                <Badge key={id} tone="hazard" shape="flame">
                  {id}
                </Badge>
              ))}
            </div>
          </Panel>
        )}

        <MetadataFooter payload={payload} />
        {loading && (
          <p className="flex items-center gap-2 text-micro text-ink-muted">
            <Spinner /> refreshing
          </p>
        )}
      </div>
    </div>
  )
}

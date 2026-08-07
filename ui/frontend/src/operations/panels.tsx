import uPlot from 'uplot'
import { useEffect, useMemo, useRef, useState } from 'react'
import { duration, integer, simClock } from '../state/format'
import { useConsole } from '../state/store'
import { Panel, ProgressBar, StatTile, StatusDot } from '../ui/primitives'

/**
 * The stall an operator most often mistakes for a hang. While a decision round
 * runs, the simulation clock is frozen on purpose, so the banner names the work
 * and shows it advancing.
 */
export function RoundBanner() {
  const round = useConsole((s) => s.snapshot?.round)
  const phase = useConsole((s) => s.session.phase)
  if (!round?.in_progress || phase !== 'running') return null
  const total = Math.max(1, round.dispatched)
  return (
    <div className="flex items-center gap-4 border-b border-status-caution/40 bg-status-caution/10 px-4 py-2">
      <span className="text-base font-semibold text-status-caution">
        Households are deciding, round {round.index}
        {round.total ? ` of ${round.total}` : ''}
      </span>
      <div className="flex-1">
        <ProgressBar value={round.resolved} max={total} tone="caution" />
      </div>
      <span className="tnum text-small text-status-caution">
        {integer(round.resolved)} of {integer(total)} answered
      </span>
      <span className="text-micro text-ink-muted">the incident clock is held while they think</span>
    </div>
  )
}

export function AccountingTiles() {
  const counts = useConsole((s) => s.snapshot?.counts)
  const live = Boolean(counts)
  const total = counts?.total ?? 0
  return (
    <div className="grid grid-cols-2 gap-2">
      <StatTile
        label="Households"
        value={live ? integer(total) : '--'}
        hint={live ? `${integer(counts!.aware)} have been warned` : 'not started'}
        tone="neutral"
      />
      <StatTile
        label="Still at home"
        value={live ? integer(counts!.waiting) : '--'}
        hint={live && total ? `${Math.round((counts!.waiting / total) * 100)}% of all households` : undefined}
        tone={live && counts!.waiting > 0 ? 'caution' : 'neutral'}
        shape="circle"
      />
      <StatTile
        label="Evacuating"
        value={live ? integer(counts!.evacuating) : '--'}
        tone="moving"
        shape="triangle"
      />
      <StatTile
        label="Arrived"
        value={live ? integer(counts!.arrived) : '--'}
        tone="nominal"
        shape="square"
      />
      <div className="col-span-2">
        <StatTile
          label="Reached by fire"
          value={live ? integer(counts!.fire_contact) : '--'}
          tone={live && counts!.fire_contact > 0 ? 'hazard' : 'neutral'}
          shape="flame"
          large={Boolean(live && counts!.fire_contact > 0)}
          hint={live && counts!.fire_contact === 0 ? 'no household has been overtaken' : undefined}
        />
      </div>
    </div>
  )
}

export function ClearanceBars() {
  const areas = useConsole((s) => s.snapshot?.areas)
  if (!areas?.length) {
    return <p className="text-small text-ink-faint">This package defines no alert areas.</p>
  }
  return (
    <ul className="space-y-2.5">
      {areas.map((area) => {
        const fraction = area.households ? area.departed / area.households : 0
        return (
          <li key={area.name}>
            <div className="flex items-baseline justify-between gap-2">
              <span className="flex items-center gap-1.5 text-small font-medium">
                <StatusDot shape="bar" tone={area.ordered ? 'caution' : 'neutral'} />
                {area.name.replace(/_/g, ' ')}
              </span>
              <span className="tnum text-micro text-ink-muted">
                {integer(area.departed)} of {integer(area.households)} left
              </span>
            </div>
            <div className="mt-1">
              <ProgressBar value={fraction} max={1} tone={fraction >= 1 ? 'nominal' : 'moving'} />
            </div>
            <p className="tnum mt-0.5 text-micro text-ink-faint">
              {area.order_t_s == null
                ? 'no order on the schedule'
                : area.ordered
                  ? `ordered at ${simClock(area.order_t_s)} by ${area.channel || 'broadcast'}`
                  : `order due at ${simClock(area.order_t_s)}`}
            </p>
          </li>
        )
      })}
    </ul>
  )
}

export function FireStatusTile() {
  const snapshot = useConsole((s) => s.snapshot)
  const preview = useConsole((s) => s.preview)
  const fires = snapshot?.fires ?? []
  const now = snapshot?.sim_t_s ?? 0
  const nextIgnition = useMemo(
    () => (preview?.fire_sources ?? []).find((source) => source.t0_s > now) ?? null,
    [preview, now],
  )
  const largest = fires.reduce((best, fire) => (fire.r_m > (best?.r_m ?? 0) ? fire : best), fires[0] ?? null)
  const nextAlert = snapshot?.alerts?.next ?? null

  return (
    <div className="space-y-2 text-small">
      <div className="flex items-baseline justify-between">
        <span className="text-ink-muted">Active fire fronts</span>
        <span className="tnum text-view font-semibold text-status-hazard">{integer(fires.length)}</span>
      </div>
      {largest && (
        <p className="tnum text-micro text-ink-muted">
          Largest front {largest.id.replace(/_/g, ' ')} at {integer(largest.r_m)} m radius
        </p>
      )}
      <p className="tnum text-micro text-ink-muted">
        {nextIgnition
          ? `Next ignition ${nextIgnition.id.replace(/_/g, ' ')} at ${simClock(nextIgnition.t0_s)}`
          : 'No further ignitions on the schedule'}
      </p>
      <div className="border-t border-ink-line pt-2">
        <p className="tnum text-micro text-ink-muted">
          {nextAlert
            ? `Next order ${nextAlert.id} at ${simClock(nextAlert.issue_time_s)} for ${nextAlert.areas
                .join(', ')
                .replace(/_/g, ' ')}`
            : 'Every scheduled order has been issued'}
        </p>
        <p className="tnum text-micro text-ink-faint">
          {integer(snapshot?.alerts?.issued?.length ?? 0)} orders issued so far
        </p>
      </div>
    </div>
  )
}

/**
 * Cumulative departures and arrivals over incident time, drawn on canvas so a
 * run of thousands of samples stays smooth.
 */
export function EvacuationCurve({ height = 170 }: { height?: number }) {
  const curve = useConsole((s) => s.curve)
  const total = useConsole((s) => s.snapshot?.counts.total ?? 0)
  const container = useRef<HTMLDivElement>(null)
  const chart = useRef<uPlot | null>(null)
  const [width, setWidth] = useState(320)

  useEffect(() => {
    if (!container.current) return
    const observer = new ResizeObserver((entries) => {
      const next = Math.max(200, Math.floor(entries[0].contentRect.width))
      setWidth(next)
    })
    observer.observe(container.current)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (!container.current) return
    const options: uPlot.Options = {
      width,
      height,
      padding: [8, 10, 0, 0],
      cursor: { drag: { x: false, y: false } },
      legend: { show: false },
      scales: { x: { time: false } },
      axes: [
        {
          stroke: '#6C7681',
          grid: { stroke: '#232B35', width: 1 },
          ticks: { stroke: '#232B35' },
          font: '11px Inter, system-ui, sans-serif',
          values: (_self, splits) => splits.map((value) => simClock(value)),
        },
        {
          stroke: '#6C7681',
          grid: { stroke: '#232B35', width: 1 },
          ticks: { stroke: '#232B35' },
          font: '11px Inter, system-ui, sans-serif',
          size: 38,
        },
      ],
      series: [
        {},
        { label: 'Left home', stroke: '#0072B2', width: 2, points: { show: false } },
        { label: 'Arrived', stroke: '#009E73', width: 2, points: { show: false } },
      ],
    }
    const instance = new uPlot(options, [[0], [0], [0]], container.current)
    chart.current = instance
    return () => {
      instance.destroy()
      chart.current = null
    }
    // The chart is rebuilt on resize, which is rare and cheap at this size.
  }, [width, height])

  useEffect(() => {
    if (!chart.current) return
    if (!curve.length) {
      chart.current.setData([[0], [0], [0]])
      return
    }
    chart.current.setData([
      curve.map((sample) => sample.t),
      curve.map((sample) => sample.departed),
      curve.map((sample) => sample.arrived),
    ])
  }, [curve])

  const last = curve[curve.length - 1]
  return (
    <div>
      <div className="mb-1.5 flex items-center gap-4 text-micro">
        <span className="flex items-center gap-1.5 text-ink-muted">
          <span className="h-0.5 w-4 bg-status-moving" /> left home
          <span className="tnum text-ink-text">{integer(last?.departed ?? 0)}</span>
        </span>
        <span className="flex items-center gap-1.5 text-ink-muted">
          <span className="h-0.5 w-4 bg-status-nominal" /> arrived
          <span className="tnum text-ink-text">{integer(last?.arrived ?? 0)}</span>
        </span>
        {total > 0 && <span className="tnum ml-auto text-ink-faint">of {integer(total)}</span>}
      </div>
      <div ref={container} className="w-full" />
      {!curve.length && (
        <p className="pt-3 text-center text-micro text-ink-faint">
          The curve fills in once the run starts stepping.
        </p>
      )}
    </div>
  )
}

/** The rail that sits beside the map during a run. */
export function KpiRail() {
  const phase = useConsole((s) => s.session.phase)
  const muted = phase === 'idle' || phase === 'preparing'
  return (
    <div className={`flex w-[340px] shrink-0 flex-col gap-3 overflow-y-auto pr-1 ${muted ? 'opacity-70' : ''}`}>
      <Panel title="Population accounting">
        <AccountingTiles />
      </Panel>
      <Panel title="Clearance by community">
        <ClearanceBars />
      </Panel>
      <Panel title="Evacuation curve">
        <EvacuationCurve />
      </Panel>
      <Panel title="Fire and orders">
        <FireStatusTile />
      </Panel>
      <ElapsedPanel />
    </div>
  )
}

function ElapsedPanel() {
  const session = useConsole((s) => s.session)
  const snapshot = useConsole((s) => s.snapshot)
  const horizon = Number(session.artifacts?.sim_end_time_s ?? session.config?.sim_end_time_s ?? 0)
  if (!horizon) return null
  const fraction = snapshot ? Math.min(1, snapshot.sim_t_s / horizon) : 0
  return (
    <Panel title="Progress through the horizon">
      <ProgressBar value={fraction} max={1} tone="nominal" height={8} />
      <p className="tnum mt-2 text-micro text-ink-muted">
        {simClock(snapshot?.sim_t_s ?? 0)} of {duration(horizon)} · {Math.round(fraction * 100)}%
      </p>
      {session.elapsed_wall_s != null && (
        <p className="tnum text-micro text-ink-faint">{duration(session.elapsed_wall_s)} of wall time so far</p>
      )}
    </Panel>
  )
}

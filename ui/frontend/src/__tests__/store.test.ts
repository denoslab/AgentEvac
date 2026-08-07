import { beforeEach, describe, expect, it } from 'vitest'
import { useConsole } from '../state/store'
import type { SessionState, SimEvent, Snapshot } from '../state/types'

function session(overrides: Partial<SessionState> = {}): SessionState {
  return {
    type: 'session',
    phase: 'running',
    detail: 'simulation running',
    run_id: 'r1',
    label: 'test run',
    config: {},
    error: null,
    anchor_clock: '15:28:00',
    elapsed_wall_s: 10,
    artifacts: {},
    stderr_tail: [],
    active: true,
    ...overrides,
  }
}

function snapshot(t: number, waiting: number, arrived: number): Snapshot {
  return {
    type: 'snapshot',
    sim_t_s: t,
    step_idx: Math.round(t * 5),
    paused: false,
    speed_target: 16,
    anchor_clock: '15:28:00',
    round: { in_progress: false, index: 1, dispatched: 0, resolved: 0, completed: 1, total: 50 },
    counts: { total: 182, waiting, evacuating: 182 - waiting - arrived, arrived, fire_contact: 0, aware: 60 },
    agents: [],
    fires: [],
    alerts: { issued: [], pending: [], next: null },
    areas: [],
  }
}

describe('telemetry ingest', () => {
  beforeEach(() => {
    useConsole.setState({
      session: session({ phase: 'idle', run_id: null, active: false }),
      snapshot: null,
      previousSnapshot: null,
      preview: null,
      events: [],
      curve: [],
      toasts: [],
      view: 'setup',
      selectedAgent: null,
    })
  })

  it('accumulates the evacuation curve from population counts', () => {
    const { ingest } = useConsole.getState()
    ingest(snapshot(100, 182, 0))
    ingest(snapshot(200, 150, 5))
    ingest(snapshot(300, 100, 40))

    const curve = useConsole.getState().curve
    expect(curve).toHaveLength(3)
    // Departed is everyone no longer at home, which is what the curve plots.
    expect(curve.map((point) => point.departed)).toEqual([0, 32, 82])
    expect(curve.map((point) => point.arrived)).toEqual([0, 5, 40])
  })

  it('ignores a snapshot that arrives out of order', () => {
    const { ingest } = useConsole.getState()
    ingest(snapshot(300, 100, 40))
    ingest(snapshot(200, 150, 5))
    expect(useConsole.getState().curve.map((point) => point.t)).toEqual([300])
  })

  it('keeps the previous snapshot so the map can interpolate between them', () => {
    const { ingest } = useConsole.getState()
    ingest(snapshot(100, 182, 0))
    ingest(snapshot(200, 150, 5))
    const state = useConsole.getState()
    expect(state.previousSnapshot?.sim_t_s).toBe(100)
    expect(state.snapshot?.sim_t_s).toBe(200)
  })

  it('drops everything from an old run when a new one starts', () => {
    const { ingest } = useConsole.getState()
    ingest(session())
    ingest(snapshot(100, 182, 0))
    ingest({ type: 'sim_event', seq: 1, event: 'departure_release', summary: 'x' } as SimEvent)
    expect(useConsole.getState().curve).toHaveLength(1)

    ingest(session({ run_id: 'r2', phase: 'preparing' }))
    const state = useConsole.getState()
    expect(state.curve).toEqual([])
    expect(state.events).toEqual([])
    expect(state.snapshot).toBeNull()
    expect(state.session.run_id).toBe('r2')
  })

  it('moves the operator to Operations when a run starts preparing', () => {
    useConsole.getState().ingest(session({ phase: 'preparing' }))
    expect(useConsole.getState().view).toBe('operations')
  })

  it('announces the end of a run once, not on every message', () => {
    const { ingest } = useConsole.getState()
    ingest(session())
    ingest(session({ phase: 'complete', detail: 'run complete', active: false }))
    ingest(session({ phase: 'complete', detail: 'run complete', active: false }))
    expect(useConsole.getState().toasts).toHaveLength(1)
    expect(useConsole.getState().toasts[0].tone).toBe('good')
  })

  it('surfaces a failure with its reason', () => {
    const { ingest } = useConsole.getState()
    ingest(session())
    ingest(session({ phase: 'failed', error: 'SUMO exited with code 1', active: false }))
    expect(useConsole.getState().toasts[0]).toMatchObject({ tone: 'bad', message: 'SUMO exited with code 1' })
  })

  it('caps the event log so a long run cannot exhaust memory', () => {
    const { ingest } = useConsole.getState()
    for (let i = 0; i < 2500; i += 1) {
      ingest({ type: 'sim_event', seq: i, event: 'llm_decision', summary: `d${i}` } as SimEvent)
    }
    const events = useConsole.getState().events
    expect(events).toHaveLength(2000)
    // The newest events are the ones kept.
    expect(events[events.length - 1].seq).toBe(2499)
  })

  it('thins the curve once it passes the render budget', () => {
    const { ingest } = useConsole.getState()
    for (let i = 1; i <= 5200; i += 1) ingest(snapshot(i, 182 - (i % 100), 0))
    const curve = useConsole.getState().curve
    expect(curve.length).toBeLessThanOrEqual(5000)
    // Thinning must keep the ends, because the last point is the current state.
    expect(curve[curve.length - 1].t).toBe(5200)
  })
})

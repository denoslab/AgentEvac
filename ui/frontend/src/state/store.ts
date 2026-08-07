import { create } from 'zustand'
import type {
  AgentFrame,
  Preview,
  SessionState,
  SimEvent,
  Snapshot,
  StreamMessage,
} from './types'

export type ConnectionState = 'connecting' | 'open' | 'retrying' | 'closed'
export type ViewName = 'setup' | 'author' | 'operations' | 'debrief'

export interface Toast {
  id: number
  message: string
  tone: 'info' | 'good' | 'warn' | 'bad'
  action?: { label: string; run: () => void }
}

/** One sample of the evacuation curve, accumulated from snapshots. */
export interface CurveSample {
  t: number
  departed: number
  arrived: number
}

/** Points beyond this are thinned, which uPlot handles without visible cost. */
const CURVE_LIMIT = 5000
const EVENT_LIMIT = 2000

const IDLE_SESSION: SessionState = {
  type: 'session',
  phase: 'idle',
  detail: 'no run started',
  run_id: null,
  label: '',
  config: {},
  error: null,
  anchor_clock: null,
  elapsed_wall_s: null,
  artifacts: {},
  stderr_tail: [],
  active: false,
}

interface ConsoleStore {
  connection: ConnectionState
  retryInSeconds: number
  session: SessionState
  snapshot: Snapshot | null
  previousSnapshot: Snapshot | null
  snapshotWall: number
  preview: Preview | null
  events: SimEvent[]
  curve: CurveSample[]
  toasts: Toast[]

  view: ViewName
  selectedAgent: string | null
  eventFilter: string
  followLatest: boolean
  layers: Record<string, boolean>
  visitedDebriefRun: string | null

  setConnection: (state: ConnectionState, retryInSeconds?: number) => void
  ingest: (message: StreamMessage) => void
  setSession: (session: SessionState) => void
  setView: (view: ViewName) => void
  selectAgent: (id: string | null) => void
  setEventFilter: (filter: string) => void
  setFollowLatest: (follow: boolean) => void
  toggleLayer: (key: string) => void
  pushToast: (message: string, tone?: Toast['tone'], action?: Toast['action']) => void
  dismissToast: (id: number) => void
  resetRunState: () => void
}

let toastSeq = 0

/** Thin a growing series in place once it passes the render budget. */
function appendCurve(series: CurveSample[], sample: CurveSample): CurveSample[] {
  if (series.length && sample.t <= series[series.length - 1].t) return series
  const next = [...series, sample]
  if (next.length <= CURVE_LIMIT) return next
  return next.filter((_, index) => index % 2 === 0 || index === next.length - 1)
}

export const useConsole = create<ConsoleStore>((set, get) => ({
  connection: 'connecting',
  retryInSeconds: 0,
  session: IDLE_SESSION,
  snapshot: null,
  previousSnapshot: null,
  snapshotWall: 0,
  preview: null,
  events: [],
  curve: [],
  toasts: [],

  view: 'setup',
  selectedAgent: null,
  eventFilter: 'all',
  followLatest: true,
  layers: { roads: true, areas: true, households: true, fires: true, destinations: true },
  visitedDebriefRun: null,

  setConnection: (connection, retryInSeconds = 0) => set({ connection, retryInSeconds }),

  setSession: (session) => set({ session }),

  setView: (view) => set({ view }),

  selectAgent: (selectedAgent) => set({ selectedAgent }),

  setEventFilter: (eventFilter) => set({ eventFilter }),

  setFollowLatest: (followLatest) => set({ followLatest }),

  toggleLayer: (key) => set((state) => ({ layers: { ...state.layers, [key]: !state.layers[key] } })),

  pushToast: (message, tone = 'info', action) =>
    set((state) => ({ toasts: [...state.toasts, { id: ++toastSeq, message, tone, action }].slice(-4) })),

  dismissToast: (id) => set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),

  resetRunState: () =>
    set({ snapshot: null, previousSnapshot: null, preview: null, events: [], curve: [], selectedAgent: null }),

  ingest: (message) => {
    switch (message.type) {
      case 'session': {
        const previous = get().session
        // A new run identifier means everything held for the old run is stale.
        if (message.run_id && message.run_id !== previous.run_id) {
          get().resetRunState()
        }
        set({ session: message })
        if (message.phase === 'preparing' && get().view === 'setup') set({ view: 'operations' })
        if (
          (message.phase === 'complete' || message.phase === 'ended_by_operator') &&
          previous.phase !== message.phase
        ) {
          get().pushToast(message.detail, 'good')
        }
        if (message.phase === 'failed' && previous.phase !== 'failed') {
          get().pushToast(message.error ?? 'the run failed', 'bad')
        }
        break
      }
      case 'snapshot': {
        const state = get()
        const departed = message.counts.total - message.counts.waiting
        set({
          previousSnapshot: state.snapshot,
          snapshot: message,
          snapshotWall: performance.now(),
          curve: appendCurve(state.curve, {
            t: message.sim_t_s,
            departed,
            arrived: message.counts.arrived,
          }),
        })
        break
      }
      case 'preview':
        set({ preview: message })
        break
      case 'sim_event': {
        const events = [...get().events, message]
        set({ events: events.length > EVENT_LIMIT ? events.slice(-EVENT_LIMIT) : events })
        break
      }
      default:
        break
    }
  },
}))

/** Agents from the latest snapshot, keyed for the interpolation loop. */
export function agentsById(snapshot: Snapshot | null): Map<string, AgentFrame> {
  const out = new Map<string, AgentFrame>()
  if (!snapshot) return out
  for (const agent of snapshot.agents) out.set(agent.id, agent)
  return out
}

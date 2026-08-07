// Shapes the backend sends. These mirror ui/backend and ui/bridge exactly, so a
// change on one side shows up as a type error on the other.

export type SessionPhase =
  | 'idle'
  | 'preparing'
  | 'running'
  | 'paused'
  | 'finishing'
  | 'complete'
  | 'ended_by_operator'
  | 'failed'

export type AgentStatus = 'waiting' | 'evacuating' | 'arrived'

export interface RunConfig {
  package: string
  scenario: string
  engine: string
  replay_run_id: string | null
  seed: number
  sim_end_time_s: number
  alert_minutes_earlier: number
  messaging: boolean
  decision_period_s: number
  initial_speed: number
  label: string
}

export interface SessionState {
  type: 'session'
  phase: SessionPhase
  detail: string
  run_id: string | null
  label: string
  config: Partial<RunConfig>
  error: string | null
  anchor_clock: string | null
  elapsed_wall_s: number | null
  artifacts: Record<string, unknown>
  stderr_tail: string[]
  active: boolean
}

export interface RoundProgress {
  in_progress: boolean
  index: number
  dispatched: number
  resolved: number
  completed: number
  total: number | null
}

export interface AgentFrame {
  id: string
  lon: number
  lat: number
  status: AgentStatus
  aware: boolean
  edge?: string | null
  fire_contact?: boolean
}

export interface FireFrame {
  id: string
  lon: number
  lat: number
  r_m: number
}

export interface AlertEventRow {
  id: string
  issue_time_s: number
  instruction: string
  areas: string[]
}

export interface AreaRow {
  name: string
  households: number
  departed: number
  arrived: number
  order_t_s: number | null
  ordered: boolean
  channel: string
}

export interface Snapshot {
  type: 'snapshot'
  sim_t_s: number
  step_idx: number
  paused: boolean
  speed_target: number | null
  anchor_clock: string | null
  round: RoundProgress
  counts: {
    total: number
    waiting: number
    evacuating: number
    arrived: number
    fire_contact: number
    aware: number
  }
  agents: AgentFrame[]
  fires: FireFrame[]
  alerts: { issued: AlertEventRow[]; pending: AlertEventRow[]; next: AlertEventRow | null }
  areas: AreaRow[]
}

export interface PreviewArea {
  name: string
  order_t_s: number
  channel: string
  households: number
  member_ids: string[]
  hull: [number, number][]
}

export interface Preview {
  type?: 'preview'
  package?: string
  households: { id: string; lon: number; lat: number; spawn_edge: string; dest_edge: string }[]
  destinations: { name: string; edge: string; lon: number; lat: number }[]
  fire_sources: {
    id: string
    lon: number
    lat: number
    t0_s: number
    r0_m: number
    growth_m_per_s: number
    max_r_m: number | null
  }[]
  areas: PreviewArea[]
  alert_events: AlertEventRow[]
  roads: GeoJSON.FeatureCollection
  bbox: [number, number, number, number] | null
  reach_bbox: [number, number, number, number] | null
}

export interface SimEvent {
  type: 'sim_event'
  seq: number
  event: string
  summary?: string
  sim_t_s?: number
  veh_id?: string
  agent_id?: string
  round?: number
  [key: string]: unknown
}

export interface SimLog {
  type: 'sim_log'
  line: string
  wall: number
}

export interface ControlAck {
  type: 'control_ack'
  action: string
  result: unknown
  wall: number
}

export type StreamMessage = SessionState | Snapshot | (Preview & { type: 'preview' }) | SimEvent | SimLog | ControlAck

// ---------------------------------------------------------------- setup data

export interface ScenarioPackage {
  id: string
  label: string
  households: number
  fire_sources: number
  destinations: string[]
  alert_waves: AlertEventRow[]
  has_alert_schedule: boolean
  net_file: string
  sumo_cfg: string
  net_exists: boolean
  net_size_mb: number
  anchor_clock: string | null
  recommended_horizon_s: number | null
  required_horizon_s: number | null
  first_ignition_s: number
  last_ignition_s: number
  description: string
  problems: string[]
  usable: boolean
  preview_ready: boolean
}

export interface PackagesPayload {
  packages: ScenarioPackage[]
  scenarios: { id: string; description: string }[]
  engines: { id: string; description: string }[]
  defaults: RunConfig
}

export interface Recording {
  run_id: string
  path: string
  size_mb: number
  modified: number
  package: string | null
  scenario: string | null
  model: string | null
  horizon_s: number | null
  params: Record<string, unknown>
  has_params: boolean
}

export interface ValidationIssue {
  field: string
  message: string
  hint: string
}

export interface ValidationResult {
  ok: boolean
  problems: ValidationIssue[]
  warnings: ValidationIssue[]
  config: RunConfig
}

export interface ScheduleRow {
  kind: 'ignition' | 'alert'
  t_s: number
  label: string
  detail: string
}

export interface OrphanRun {
  run_id: string
  run_dir: string
  port: number
  label: string | null
  config: Partial<RunConfig> | null
  launched_wall: number | null
  anchor_clock: string | null
  phase: string
  sim_t_s: number | null
}

export interface HealthPayload {
  version: string
  uptime_s: number
  environment: {
    sumo_home: string
    sumo_home_exists: boolean
    sumo_binary: string | null
    openai_key: boolean
    python: string
    simulator_python: string | null
    simulator_python_hint: string | null
  }
  ready: boolean
  frontend_built: boolean
  preview_assets: string[]
  orphan_run: OrphanRun | null
}

// -------------------------------------------------------------- debrief data

export interface RunRecord {
  run_id: string
  metrics_path: string
  modified: number
  campaign: string
  label: string
  package: string | null
  scenario: string | null
  agent_type: string | null
  seed: number | null
  horizon_s: number | null
  alert_offset_s: number | null
  total_agents: number | null
  arrived: number | null
  departed: number | null
  from_console: boolean
}

export interface MetricsPayload {
  run_id: string
  record: RunRecord
  metrics: Record<string, any>
  curve: { departures: [number, number][]; arrivals: [number, number][] }
  artifacts: Record<string, string | null>
}

// --- Authoring a package from a map selection ---

/** One building as the authoring map draws and hit-tests it. */
export interface AuthorBuilding {
  id: string
  lon: number
  lat: number
  /** Simulation coordinates, which is what the written package records. */
  x?: number
  y?: number
  /** The road a household here would spawn onto, null when none is close enough. */
  edge: string | null
  edge_dist_m: number | null
  poly?: [number, number][]
}

export interface BuildingsPayload {
  package: string
  count: number
  buildings: AuthorBuilding[]
}

/** A fire origin as the authoring view holds it, in simulation coordinates. */
export interface DraftFire {
  id: string
  x: number
  y: number
  t0: number
  r0: number
  growth_m_per_s: number
  max_r_m?: number | null
}

export interface DraftArea {
  name: string
  label?: string
  building_ids: string[]
}

export interface DraftAlertEvent {
  id: string
  issue_time_s: number
  areas: string[]
  instruction: string
  channel: string
  hazard_text: string
  routing_text?: string | null
}

/** The body POSTed to create a package. */
export interface PackageDraft {
  id: string
  label: string
  description: string
  source_package: string
  households: { building_id: string; count: number }[]
  fires: DraftFire[]
  alert_areas?: DraftArea[]
  alert_events?: DraftAlertEvent[]
}

export interface DraftSummary {
  package: string
  source_package: string
  households: number
  agents: number
  edges: number
  fire_sources: number
  alert_areas: number
  alert_events: number
}

export interface DraftValidation {
  ok: boolean
  problems: ValidationIssue[]
  warnings: ValidationIssue[]
  summary: DraftSummary
  package?: string
  path?: string
  preview_ready?: boolean
}

import type {
  BuildingsPayload,
  DraftValidation,
  HealthPayload,
  MetricsPayload,
  PackagesPayload,
  PackageDraft,
  Preview,
  Recording,
  RunConfig,
  RunRecord,
  ScheduleRow,
  ScenarioPackage,
  SessionState,
  ValidationResult,
} from './types'

export class ApiError extends Error {
  status: number
  payload: unknown
  constructor(message: string, status: number, payload: unknown) {
    super(message)
    this.status = status
    this.payload = payload
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    })
  } catch (cause) {
    throw new ApiError('the console backend is not reachable', 0, cause)
  }
  const text = await response.text()
  let payload: unknown = null
  if (text) {
    try {
      payload = JSON.parse(text)
    } catch {
      payload = text
    }
  }
  if (!response.ok) {
    const detail =
      (payload as { detail?: string; error?: string })?.detail ??
      (payload as { error?: string })?.error ??
      response.statusText
    throw new ApiError(detail, response.status, payload)
  }
  return payload as T
}

export const api = {
  health: () => request<HealthPayload>('/api/health'),
  packages: () => request<PackagesPayload>('/api/packages'),
  packageSchedule: (id: string) =>
    request<{ package: ScenarioPackage; schedule: ScheduleRow[] }>(
      `/api/packages/${encodeURIComponent(id)}/schedule`,
    ),
  packagePreview: (id: string) => request<Preview>(`/api/packages/${encodeURIComponent(id)}/preview`),
  packageBuildings: (id: string) =>
    request<BuildingsPayload>(`/api/packages/${encodeURIComponent(id)}/buildings`),
  validatePackage: (draft: PackageDraft) =>
    request<DraftValidation>('/api/packages/validate', {
      method: 'POST',
      body: JSON.stringify(draft),
    }),
  createPackage: (draft: PackageDraft) =>
    request<DraftValidation>('/api/packages', {
      method: 'POST',
      body: JSON.stringify(draft),
    }),
  recordings: () => request<{ recordings: Recording[] }>('/api/recordings'),
  validate: (config: Partial<RunConfig>) =>
    request<ValidationResult>('/api/runs/validate', {
      method: 'POST',
      body: JSON.stringify(config),
    }),
  launch: (config: Partial<RunConfig>) =>
    request<{ session: SessionState; warnings: unknown[] }>('/api/runs', {
      method: 'POST',
      body: JSON.stringify(config),
    }),
  currentRun: () => request<SessionState>('/api/runs/current'),
  quit: (endRun = false) =>
    request<{ ok: boolean; ended_run: boolean }>('/api/quit', {
      method: 'POST',
      body: JSON.stringify({ end_run: endRun }),
    }),
  adoptOrphan: () => request<{ session: SessionState }>('/api/runs/adopt', { method: 'POST' }),
  discardOrphan: () => request<{ ok: boolean }>('/api/runs/discard-orphan', { method: 'POST' }),
  control: (action: string, value?: unknown) =>
    request<{ ok: boolean; session: SessionState }>('/api/runs/current/control', {
      method: 'POST',
      body: JSON.stringify({ action, value }),
    }),
  agentDetail: (agentId: string) =>
    request<Record<string, any>>(`/api/runs/current/agents/${encodeURIComponent(agentId)}`),
  history: (limit = 200) => request<{ runs: RunRecord[] }>(`/api/history?limit=${limit}`),
  runMetrics: (runId: string) =>
    request<MetricsPayload>(`/api/history/${encodeURIComponent(runId)}/metrics`),
  exportUrl: (runId: string) => `/api/history/${encodeURIComponent(runId)}/export`,
  fileUrl: (repoRelativePath: string) =>
    `/api/files/${repoRelativePath.split('/').map(encodeURIComponent).join('/')}`,
}

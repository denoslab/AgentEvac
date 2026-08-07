import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, ApiError } from '../state/api'
import { integer, simClock } from '../state/format'
import { useConsole } from '../state/store'
import type { AuthorBuilding, DraftFire, DraftValidation, Preview } from '../state/types'
import { Badge, Button, EmptyState, Panel, StatTile } from '../ui/primitives'
import { AuthorMap, lonLatToSim, type AuthorTool } from './AuthorMap'
import {
  addAreaMembers,
  addToSelection,
  buildDraft,
  buildingsInBox,
  fireRadiusAt,
  MAX_AGENTS_PER_BUILDING,
  newFire,
  packageIdProblem,
  pruneAreaMembers,
  removeAreaMembers,
  removeFromSelection,
  setAllCounts,
  setCount,
  suggestPackageId,
  toggleAreaMember,
  toggleBuilding,
  totals,
  unspawnableInBox,
  type Box,
  type Household,
} from './selection'

const TOOLS: { id: AuthorTool; label: string; hint: string }[] = [
  { id: 'households', label: 'Households', hint: 'Drag a box over the buildings that should evacuate.' },
  { id: 'count', label: 'Agents', hint: 'Click a household to set how many agents it holds.' },
  { id: 'area', label: 'Alert area', hint: 'Drag a box over the households the order covers.' },
  { id: 'fire', label: 'Fire origins', hint: 'Click where a fire starts, then set how it grows.' },
]

/**
 * Setting one building's agent count, anchored where the building is.
 *
 * Most buildings are houses holding one vehicle. A school or a care home holds many, and
 * finding that one building among several hundred is only practical on the map itself,
 * so the editor comes to the building rather than the other way round.
 */
function CountStepper({
  buildingId,
  count,
  position,
  onChange,
  onClose,
}: {
  buildingId: string
  count: number
  position: { x: number; y: number }
  onChange: (next: number) => void
  onClose: () => void
}) {
  const step = (delta: number) => onChange(count + delta)
  return (
    <div
      className="panel absolute z-20 w-44 -translate-x-1/2 -translate-y-full p-2 shadow-lg"
      style={{ left: position.x, top: position.y - 12 }}
      onPointerDown={(event) => event.stopPropagation()}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-micro text-ink-faint" title={buildingId}>
          building {buildingId}
        </span>
        <button type="button" className="text-micro text-ink-faint hover:text-ink-text" onClick={onClose}>
          close
        </button>
      </div>
      <div className="mt-1.5 flex items-center gap-1">
        <button type="button" className="btn btn-ghost px-2" onClick={() => step(-1)} disabled={count <= 1}>
          -
        </button>
        <input
          className="input tnum w-full text-center"
          type="number"
          min={1}
          max={MAX_AGENTS_PER_BUILDING}
          value={count}
          onChange={(event) => onChange(Number(event.target.value))}
        />
        <button
          type="button"
          className="btn btn-ghost px-2"
          onClick={() => step(1)}
          disabled={count >= MAX_AGENTS_PER_BUILDING}
        >
          +
        </button>
      </div>
      <div className="mt-1.5 flex gap-1">
        {[1, 2, 5, 20].map((preset) => (
          <button
            key={preset}
            type="button"
            className={`flex-1 rounded border px-1 py-0.5 text-micro ${
              count === preset ? 'border-accent text-ink-text' : 'border-ink-line text-ink-muted'
            }`}
            onClick={() => onChange(preset)}
          >
            {preset}
          </button>
        ))}
      </div>
    </div>
  )
}

function Field({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: React.ReactNode
}) {
  return (
    <label className="block">
      <span className="text-small font-medium text-ink-muted">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-micro text-ink-faint">{hint}</span>}
    </label>
  )
}

function IssueList({ issues, tone }: { issues: { field: string; message: string; hint: string }[]; tone: 'bad' | 'warn' }) {
  if (issues.length === 0) return null
  return (
    <ul className="space-y-1.5">
      {issues.map((issue, i) => (
        <li key={`${issue.field}-${i}`} className="text-micro">
          <span className={tone === 'bad' ? 'text-status-hazard' : 'text-status-caution'}>
            {issue.message}
          </span>
          <span className="block text-ink-faint">{issue.hint}</span>
        </li>
      ))}
    </ul>
  )
}

function FireEditor({
  fire,
  onChange,
  onRemove,
}: {
  fire: DraftFire
  onChange: (next: DraftFire) => void
  onRemove: () => void
}) {
  const num = (key: keyof DraftFire) => (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = Number(event.target.value)
    onChange({ ...fire, [key]: Number.isFinite(value) ? value : 0 })
  }
  return (
    <div className="rounded border border-line bg-surface-sunken p-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-small font-medium">{fire.id}</span>
        <Button variant="ghost" onClick={onRemove}>
          Remove
        </Button>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-2">
        <Field label="Starts at (s)">
          <input className="input tnum" type="number" value={fire.t0} onChange={num('t0')} />
        </Field>
        <Field label="Initial radius (m)">
          <input className="input tnum" type="number" value={fire.r0} onChange={num('r0')} />
        </Field>
        <Field label="Growth (m/s)">
          <input className="input tnum" type="number" step="0.1" value={fire.growth_m_per_s} onChange={num('growth_m_per_s')} />
        </Field>
        <Field label="Radius cap (m)">
          <input
            className="input tnum"
            type="number"
            value={fire.max_r_m ?? ''}
            placeholder="none"
            onChange={(event) =>
              onChange({ ...fire, max_r_m: event.target.value === '' ? null : Number(event.target.value) })
            }
          />
        </Field>
      </div>
    </div>
  )
}

/**
 * Authoring a scenario package by drawing on the map.
 *
 * Three tools over one map. Households are the buildings that evacuate, the alert area is
 * the buildings an order covers, and fire origins are placed by click. The package is
 * validated on the backend as the draft changes, so the operator sees a named problem
 * while there is still something to change, and creating it never writes over anything.
 */
export function AuthorView({ onDone }: { onDone: (packageId: string) => void }) {
  const pushToast = useConsole((s) => s.pushToast)

  const [sourcePackage, setSourcePackage] = useState<string>('')
  const [sources, setSources] = useState<string[]>([])
  const [buildings, setBuildings] = useState<AuthorBuilding[]>([])
  const [roads, setRoads] = useState<GeoJSON.FeatureCollection | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)

  const [tool, setTool] = useState<AuthorTool>('households')
  const [subtractive, setSubtractive] = useState(false)
  const [households, setHouseholds] = useState<Household[]>([])
  const [areaMembers, setAreaMembers] = useState<string[]>([])
  const [fires, setFires] = useState<DraftFire[]>([])
  const [selectedFire, setSelectedFire] = useState<string | null>(null)
  const [anchorId, setAnchorId] = useState<string | null>(null)
  const [anchorPos, setAnchorPos] = useState<{ x: number; y: number } | null>(null)
  const [previewTimeS, setPreviewTimeS] = useState(1800)

  const [label, setLabel] = useState('')
  const [packageId, setPackageId] = useState('')
  const [idTouched, setIdTouched] = useState(false)
  const [description, setDescription] = useState('')
  const [areaName, setAreaName] = useState('ordered_area')
  const [areaOrderTimeS, setAreaOrderTimeS] = useState(1800)
  const [areaHazardText, setAreaHazardText] = useState('Evacuate immediately.')

  const [staleBundle, setStaleBundle] = useState(false)
  const [validation, setValidation] = useState<DraftValidation | null>(null)
  const [creating, setCreating] = useState(false)

  const index = useMemo(() => new Map(buildings.map((b) => [b.id, b])), [buildings])
  const householdIds = useMemo(() => new Set(households.map((h) => h.building_id)), [households])
  const areaIds = useMemo(() => new Set(areaMembers), [areaMembers])
  const counts = useMemo(
    () => new Map(households.map((h) => [h.building_id, h.count])),
    [households],
  )
  const raised = useMemo(() => households.filter((h) => h.count > 1).length, [households])

  // The area is a subset of the households by construction, so removing a household
  // removes it from the area in the same breath.
  useEffect(() => {
    setAreaMembers((current) => {
      const pruned = pruneAreaMembers(current, householdIds)
      return pruned.length === current.length ? current : pruned
    })
  }, [householdIds])
  const counted = useMemo(() => totals(households, index), [households, index])

  // Holding shift turns a drag into a removal, which is the fastest way to trim a
  // selection that overshot.
  useEffect(() => {
    const down = (event: KeyboardEvent) => event.key === 'Shift' && setSubtractive(true)
    const up = (event: KeyboardEvent) => event.key === 'Shift' && setSubtractive(false)
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
    }
  }, [])

  // The list of packages a draft can inherit its network and shelters from.
  useEffect(() => {
    api
      .packages()
      .then((payload) => {
        const usable = payload.packages.filter((p) => p.usable && p.preview_ready).map((p) => p.id)
        setSources(usable)
        if (usable.length > 0) setSourcePackage((current) => current || usable[0])
      })
      .catch(() => setLoadError('The console backend is not answering.'))
  }, [])

  // The building layer and the basemap for whichever source package is chosen.
  useEffect(() => {
    if (!sourcePackage) return
    let cancelled = false
    setLoading(true)
    setLoadError(null)
    Promise.all([api.packageBuildings(sourcePackage), api.packagePreview(sourcePackage)])
      .then(([layer, preview]: [{ buildings: AuthorBuilding[] }, Preview]) => {
        if (cancelled) return
        setBuildings(layer.buildings)
        setRoads(preview.roads ?? null)
        // Simulation coordinates arrived in the bundle after the first ones were built,
        // and a fire cannot be placed without them.
        setStaleBundle(layer.buildings.length > 0 && layer.buildings[0].x == null)
        setHouseholds([])
        setAreaMembers([])
        setFires([])
      })
      .catch((error: unknown) => {
        if (cancelled) return
        setBuildings([])
        setRoads(null)
        setLoadError(
          error instanceof ApiError && error.status === 404
            ? `${sourcePackage} has no building layer. Run python -m ui.tools.build_map_assets ${sourcePackage}.`
            : 'The building layer could not be loaded.',
        )
      })
      .finally(() => !cancelled && setLoading(false))
    return () => {
      cancelled = true
    }
  }, [sourcePackage])

  const effectiveId = idTouched ? packageId : suggestPackageId(label)
  const idProblem = packageIdProblem(effectiveId)

  const draft = useMemo(
    () =>
      buildDraft({
        id: effectiveId,
        label,
        description,
        sourcePackage,
        households,
        areaName,
        areaMembers,
        areaOrderTimeS,
        areaChannel: 'broadcast',
        areaInstruction: 'evacuate_now',
        areaHazardText,
        fires,
      }),
    [effectiveId, label, description, sourcePackage, households, areaName, areaMembers,
     areaOrderTimeS, areaHazardText, fires],
  )

  // Validation is asked for as the draft settles, so a problem shows up while the
  // operator is still looking at the thing that caused it.
  const validateTimer = useRef<number | null>(null)
  useEffect(() => {
    if (!sourcePackage || households.length === 0) {
      setValidation(null)
      return
    }
    if (validateTimer.current) window.clearTimeout(validateTimer.current)
    validateTimer.current = window.setTimeout(() => {
      api
        .validatePackage(draft)
        .then(setValidation)
        .catch(() => setValidation(null))
    }, 400)
    return () => {
      if (validateTimer.current) window.clearTimeout(validateTimer.current)
    }
  }, [draft, sourcePackage, households.length])

  // ------------------------------------------------------------------ drawing
  const onBox = useCallback(
    (box: Box, subtract: boolean) => {
      const picked = buildingsInBox(buildings, box)
      if (tool === 'households') {
        const dropped = unspawnableInBox(buildings, box).length
        setHouseholds((current) =>
          subtract ? removeFromSelection(current, picked) : addToSelection(current, picked),
        )
        if (!subtract && dropped > 0) {
          pushToast(`${dropped} buildings in that box sit too far from a road to spawn`, 'warn')
        }
      } else if (tool === 'area') {
        setAreaMembers((current) =>
          subtract
            ? removeAreaMembers(current, picked)
            : addAreaMembers(current, picked, householdIds),
        )
        const skipped = picked.filter((b) => !householdIds.has(b.id)).length
        if (!subtract && skipped > 0) {
          pushToast(
            `${skipped} buildings in that box are not households, so the order skips them`,
            'warn',
          )
        }
      }
    },
    [buildings, tool, householdIds, pushToast],
  )

  const onBuildingClick = useCallback(
    (buildingId: string) => {
      const building = index.get(buildingId)
      if (!building) return
      if (tool === 'households') setHouseholds((current) => toggleBuilding(current, building))
      else if (tool === 'area') {
        if (!householdIds.has(buildingId)) {
          pushToast('Only a household can be ordered. Select it under Households first.', 'warn')
          return
        }
        setAreaMembers((current) => toggleAreaMember(current, buildingId, householdIds))
      }
      else if (tool === 'count') {
        // Only a household holds agents, so clicking anything else says so instead of
        // opening an editor that could not change anything.
        if (!households.some((h) => h.building_id === buildingId)) {
          pushToast('That building is not a household yet. Select it first.', 'warn')
          return
        }
        setAnchorId(buildingId)
      }
    },
    [index, tool, households, householdIds, pushToast],
  )

  const onPlaceFire = useCallback(
    (lon: number, lat: number) => {
      const sim = lonLatToSim(lon, lat, buildings)
      if (!sim) {
        pushToast(
          `Rebuild this package's map bundle first: python -m ui.tools.build_map_assets ${sourcePackage} --force`,
          'warn',
        )
        return
      }
      setFires((current) => {
        const fire = newFire(current.length, sim.x, sim.y)
        setSelectedFire(fire.id)
        return [...current, fire]
      })
    },
    [buildings, pushToast, sourcePackage],
  )

  const create = async () => {
    setCreating(true)
    try {
      const result = await api.createPackage(draft)
      pushToast(`Created ${result.package}. Build its map bundle to see it in Setup.`, 'good')
      onDone(result.package ?? effectiveId)
    } catch (error) {
      if (error instanceof ApiError && error.payload) {
        setValidation(error.payload as DraftValidation)
        pushToast(
          error.status === 409 ? 'That package name is already taken' : 'The package was not created',
          'bad',
        )
      } else {
        pushToast('The package was not created', 'bad')
      }
    } finally {
      setCreating(false)
    }
  }

  const blocked = Boolean(idProblem) || households.length === 0 || fires.length === 0 || !validation?.ok
  const activeTool = TOOLS.find((t) => t.id === tool)

  return (
    <div className="grid h-full min-h-0 grid-cols-[380px_minmax(0,1fr)] gap-3 p-3">
      <div className="flex min-h-0 flex-col gap-3 overflow-auto">
        <Panel title="Package" bodyClassName="p-0">
          <div className="space-y-3 p-3">
            <Field label="Source package" hint="Its road network, shelters, and routes are inherited.">
              <select
                className="input"
                value={sourcePackage}
                onChange={(event) => setSourcePackage(event.target.value)}
              >
                {sources.length === 0 && <option value="">no package with a map bundle</option>}
                {sources.map((id) => (
                  <option key={id} value={id}>
                    {id}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Name shown in Setup">
              <input
                className="input"
                value={label}
                placeholder="Westwood ignition area"
                onChange={(event) => setLabel(event.target.value)}
              />
            </Field>
            <Field label="Directory name" hint={idProblem ?? 'Created under configs/, never overwriting.'}>
              <input
                className="input"
                value={effectiveId}
                onChange={(event) => {
                  setIdTouched(true)
                  setPackageId(event.target.value)
                }}
              />
            </Field>
            <Field label="Note" hint="Written into the package README.">
              <textarea
                className="input h-16 resize-none"
                value={description}
                onChange={(event) => setDescription(event.target.value)}
              />
            </Field>
          </div>
        </Panel>

        <Panel title="Drawing" bodyClassName="p-0">
          <div className="space-y-3 p-3">
            {staleBundle && (
              <p className="text-micro text-status-caution">
                This package&rsquo;s map bundle predates simulation coordinates, so fire origins
                cannot be placed. Rebuild it with python -m ui.tools.build_map_assets{' '}
                {sourcePackage} --force
              </p>
            )}
            <div className="flex gap-1">
              {TOOLS.map((entry) => (
                <button
                  key={entry.id}
                  type="button"
                  disabled={entry.id === 'fire' && staleBundle}
                  onClick={() => {
                    setTool(entry.id)
                    setAnchorId(null)
                  }}
                  className={`card-choice flex-1 text-center ${
                    tool === entry.id ? 'card-choice-on' : 'card-choice-off'
                  }`}
                >
                  <span className="text-small font-medium">{entry.label}</span>
                </button>
              ))}
            </div>
            <p className="text-micro text-ink-faint">
              {activeTool?.hint} Hold shift and drag to remove. Pan with the right or middle
              button, zoom with the wheel.
            </p>

            <div className="grid grid-cols-3 gap-2">
              <StatTile label="Buildings" value={integer(counted.buildings)} />
              <StatTile label="Agents" value={integer(counted.agents)} />
              <StatTile label="Roads" value={integer(counted.roads)} />
            </div>

            {tool === 'count' && (
              <p className="text-micro text-ink-faint">
                {raised === 0
                  ? 'Every household holds one agent. Click one on the map to raise it.'
                  : `${integer(raised)} buildings hold more than one agent, ringed in white on the map.`}
              </p>
            )}

            {tool === 'households' && households.length > 0 && (
              <Field
                label="Set every household at once"
                hint="Overwrites any per-building count already set. Use the Agents tool for one building."
              >
                <input
                  className="input tnum"
                  type="number"
                  min={1}
                  max={MAX_AGENTS_PER_BUILDING}
                  defaultValue={1}
                  onChange={(event) => setHouseholds((c) => setAllCounts(c, Number(event.target.value)))}
                />
              </Field>
            )}

            {households.length > 0 && (
              <Button variant="ghost" onClick={() => setHouseholds([])}>
                Clear households
              </Button>
            )}
          </div>
        </Panel>

        {tool === 'area' && (
          <Panel title="Alert area" bodyClassName="p-0">
            <div className="space-y-3 p-3">
              <p className="text-micro text-ink-faint">
                {integer(areaMembers.length)} of {integer(households.length)} households ordered,
                shown orange. Blue households are selected but not ordered. The order is scoped to
                the roads those households sit on.
              </p>
              {households.length === 0 && (
                <p className="text-micro text-status-caution">
                  Select households first. An order only reaches buildings that hold agents.
                </p>
              )}
              <Field label="Area name">
                <input className="input" value={areaName} onChange={(e) => setAreaName(e.target.value)} />
              </Field>
              <Field label="Order issued at (s)" hint={simClock(areaOrderTimeS)}>
                <input
                  className="input tnum"
                  type="number"
                  value={areaOrderTimeS}
                  onChange={(e) => setAreaOrderTimeS(Number(e.target.value) || 0)}
                />
              </Field>
              <Field label="Message">
                <textarea
                  className="input h-16 resize-none"
                  value={areaHazardText}
                  onChange={(e) => setAreaHazardText(e.target.value)}
                />
              </Field>
              <div className="flex gap-2">
                <Button
                  variant="ghost"
                  disabled={households.length === 0 || areaMembers.length === households.length}
                  onClick={() => setAreaMembers(households.map((h) => h.building_id))}
                >
                  Order all households
                </Button>
                {areaMembers.length > 0 && (
                  <Button variant="ghost" onClick={() => setAreaMembers([])}>
                    Clear area
                  </Button>
                )}
              </div>
            </div>
          </Panel>
        )}

        {tool === 'fire' && (
          <Panel title="Fire origins" bodyClassName="p-0">
            <div className="space-y-3 p-3">
              {fires.length === 0 && (
                <p className="text-micro text-ink-faint">Click the map to place the first origin.</p>
              )}
              <Field label={`Front shown at ${simClock(previewTimeS)}`}>
                <input
                  type="range"
                  min={0}
                  max={28800}
                  step={60}
                  value={previewTimeS}
                  className="w-full"
                  onChange={(event) => setPreviewTimeS(Number(event.target.value))}
                />
              </Field>
              <div className="space-y-2">
                {fires.map((fire) => (
                  <div key={fire.id} onFocus={() => setSelectedFire(fire.id)}>
                    <FireEditor
                      fire={fire}
                      onChange={(next) =>
                        setFires((current) => current.map((f) => (f.id === fire.id ? next : f)))
                      }
                      onRemove={() => setFires((current) => current.filter((f) => f.id !== fire.id))}
                    />
                    <p className="tnum mt-1 text-micro text-ink-faint">
                      radius {integer(fireRadiusAt(fire, previewTimeS))} m at {simClock(previewTimeS)}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </Panel>
        )}

        <Panel title="Before it is created" bodyClassName="p-0">
          <div className="space-y-3 p-3">
            {validation && (
              <>
                <IssueList issues={validation.problems} tone="bad" />
                <IssueList issues={validation.warnings} tone="warn" />
                {validation.ok && validation.problems.length === 0 && (
                  <p className="text-micro text-status-good">Ready to create.</p>
                )}
              </>
            )}
            {households.length === 0 && (
              <p className="text-micro text-ink-faint">Select at least one household.</p>
            )}
            {fires.length === 0 && (
              <p className="text-micro text-ink-faint">Place at least one fire origin.</p>
            )}
            <Button variant="primary" onClick={create} busy={creating} disabled={blocked}>
              Create package
            </Button>
            <p className="text-micro text-ink-faint">
              Creating writes a new directory under configs/. An existing package is never changed.
            </p>
          </div>
        </Panel>
      </div>

      <Panel
        title="Map"
        bodyClassName="min-h-0 flex-1 p-0"
        action={
          <div className="flex items-center gap-2">
            {subtractive && <Badge tone="caution">removing</Badge>}
            <span className="text-micro text-ink-faint">{integer(buildings.length)} buildings</span>
          </div>
        }
      >
        <div className="relative h-full w-full">
          {loadError ? (
            <div className="flex h-full items-center justify-center p-6">
              <EmptyState title="No building layer" detail={loadError} />
            </div>
          ) : loading ? (
            <div className="flex h-full items-center justify-center text-ink-muted">
              Loading the building layer
            </div>
          ) : (
            <AuthorMap
              buildings={buildings}
              roads={roads}
              householdIds={householdIds}
              areaIds={areaIds}
              counts={counts}
              anchorId={tool === 'count' ? anchorId : null}
              onAnchorMove={setAnchorPos}
              fires={fires}
              selectedFire={selectedFire}
              previewTimeS={previewTimeS}
              tool={tool}
              subtractive={subtractive}
              onBox={onBox}
              onBuildingClick={onBuildingClick}
              onPlaceFire={onPlaceFire}
              onSelectFire={setSelectedFire}
            />
          )}
          {tool === 'count' && anchorId && anchorPos && (
            <CountStepper
              buildingId={anchorId}
              count={counts.get(anchorId) ?? 1}
              position={anchorPos}
              onChange={(next) => setHouseholds((current) => setCount(current, anchorId, next))}
              onClose={() => setAnchorId(null)}
            />
          )}
        </div>
      </Panel>
    </div>
  )
}

export { setCount }

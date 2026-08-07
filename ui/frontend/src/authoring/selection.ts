import type { AuthorBuilding, DraftFire, PackageDraft } from '../state/types'

// Selecting households, selecting the area an order covers, and placing fire origins are
// all the same shape of decision, so the rules live here as plain functions the view
// calls. Nothing in this file touches MapLibre or React, which is what lets the parts an
// operator can get wrong be tested directly.

/** A drawn box, in longitude and latitude, in the order the two corners were clicked. */
export interface Box {
  lon1: number
  lat1: number
  lon2: number
  lat2: number
}

/** A household in the draft, meaning one selected building and how many agents it holds. */
export interface Household {
  building_id: string
  count: number
}

export const DEFAULT_AGENTS_PER_BUILDING = 1
export const MAX_AGENTS_PER_BUILDING = 50

/** Normalise a drawn box, since an operator may drag from any corner. */
export function normaliseBox(box: Box): { minLon: number; minLat: number; maxLon: number; maxLat: number } {
  return {
    minLon: Math.min(box.lon1, box.lon2),
    minLat: Math.min(box.lat1, box.lat2),
    maxLon: Math.max(box.lon1, box.lon2),
    maxLat: Math.max(box.lat1, box.lat2),
  }
}

/**
 * Buildings a drawn box selects.
 *
 * Selection is by centroid, so a building is in or out as a whole and an operator never
 * has to reason about a footprint straddling the edge of the box. Buildings with no road
 * to spawn onto are excluded here, because a household cannot be placed on one, and
 * leaving them in would mean the count on screen disagreed with the package written.
 */
export function buildingsInBox(buildings: AuthorBuilding[], box: Box): AuthorBuilding[] {
  const { minLon, minLat, maxLon, maxLat } = normaliseBox(box)
  return buildings.filter(
    (b) => Boolean(b.edge) && b.lon >= minLon && b.lon <= maxLon && b.lat >= minLat && b.lat <= maxLat,
  )
}

/** Buildings inside the box that cannot spawn, so the view can say how many were dropped. */
export function unspawnableInBox(buildings: AuthorBuilding[], box: Box): AuthorBuilding[] {
  const { minLon, minLat, maxLon, maxLat } = normaliseBox(box)
  return buildings.filter(
    (b) => !b.edge && b.lon >= minLon && b.lon <= maxLon && b.lat >= minLat && b.lat <= maxLat,
  )
}

/**
 * Add a box's buildings to the current households, keeping any count already edited.
 *
 * Boxes accumulate, so an operator can build a selection out of several draws without
 * losing the counts they set on an earlier one.
 */
export function addToSelection(current: Household[], picked: AuthorBuilding[]): Household[] {
  const byId = new Map(current.map((h) => [h.building_id, h]))
  for (const building of picked) {
    if (!byId.has(building.id)) {
      byId.set(building.id, { building_id: building.id, count: DEFAULT_AGENTS_PER_BUILDING })
    }
  }
  return [...byId.values()]
}

/** Remove a box's buildings from the current households. */
export function removeFromSelection(current: Household[], picked: AuthorBuilding[]): Household[] {
  const drop = new Set(picked.map((b) => b.id))
  return current.filter((h) => !drop.has(h.building_id))
}

/** Add one building if absent, remove it if present, which is what a click does. */
export function toggleBuilding(current: Household[], building: AuthorBuilding): Household[] {
  if (!building.edge) return current
  const present = current.some((h) => h.building_id === building.id)
  return present
    ? current.filter((h) => h.building_id !== building.id)
    : [...current, { building_id: building.id, count: DEFAULT_AGENTS_PER_BUILDING }]
}

/** Set one building's agent count, clamped to what the backend will accept. */
export function setCount(current: Household[], buildingId: string, count: number): Household[] {
  const clamped = Math.max(1, Math.min(MAX_AGENTS_PER_BUILDING, Math.round(count || 0)))
  return current.map((h) => (h.building_id === buildingId ? { ...h, count: clamped } : h))
}

/** Set every selected building's count at once, which is the usual way to scale a draft. */
export function setAllCounts(current: Household[], count: number): Household[] {
  const clamped = Math.max(1, Math.min(MAX_AGENTS_PER_BUILDING, Math.round(count || 0)))
  return current.map((h) => ({ ...h, count: clamped }))
}

// An alert area only ever covers buildings that are already households. A building with
// no agents cannot be evacuated, so ordering it changes nothing, and its road would be
// pulled into the ordered area anyway, which would over-scope the order onto households
// that were never selected. Holding the area to a subset of the households also means an
// orange building on the map is necessarily a selected household, so the two states can
// never be confused for one another.

/** Toggle one household's membership of the alert area. */
export function toggleAreaMember(
  current: string[],
  buildingId: string,
  households: Set<string>,
): string[] {
  if (!households.has(buildingId)) return current
  return current.includes(buildingId)
    ? current.filter((id) => id !== buildingId)
    : [...current, buildingId]
}

/** Add a box's households to the alert area, without duplicating any already in it. */
export function addAreaMembers(
  current: string[],
  picked: AuthorBuilding[],
  households: Set<string>,
): string[] {
  const seen = new Set(current)
  const out = [...current]
  for (const building of picked) {
    if (households.has(building.id) && !seen.has(building.id)) {
      seen.add(building.id)
      out.push(building.id)
    }
  }
  return out
}

/** Remove a box's buildings from the alert area. */
export function removeAreaMembers(current: string[], picked: AuthorBuilding[]): string[] {
  const drop = new Set(picked.map((b) => b.id))
  return current.filter((id) => !drop.has(id))
}

/**
 * Drop area members that are no longer households.
 *
 * Deselecting a household has to take it out of the area too, otherwise the area would
 * quietly keep ordering a building that no longer holds anyone.
 */
export function pruneAreaMembers(current: string[], households: Set<string>): string[] {
  return current.filter((id) => households.has(id))
}

export interface SelectionTotals {
  buildings: number
  agents: number
  roads: number
}

/** What the header counts while an operator is drawing. */
export function totals(households: Household[], index: Map<string, AuthorBuilding>): SelectionTotals {
  const roads = new Set<string>()
  let agents = 0
  for (const household of households) {
    agents += household.count
    const edge = index.get(household.building_id)?.edge
    if (edge) roads.add(edge)
  }
  return { buildings: households.length, agents, roads: roads.size }
}

/**
 * A package name the backend will take.
 *
 * The same rule the backend enforces, applied as the operator types, so a long draft is
 * never rejected at the end for something that could have been said at the start.
 */
export function packageIdProblem(id: string): string | null {
  if (!id) return 'Give the package a name.'
  if (!/^[a-z0-9]/.test(id)) return 'Start the name with a lowercase letter or a digit.'
  if (id.length < 3) return 'Use at least three characters.'
  if (id.length > 64) return 'Keep the name to 64 characters or fewer.'
  if (!/^[a-z0-9_]+$/.test(id)) return 'Use lowercase letters, digits, and underscores only.'
  return null
}

/** Turn a typed label into a usable package name, which is what the field suggests. */
export function suggestPackageId(label: string): string {
  return label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 64)
}

export interface DraftInput {
  id: string
  label: string
  description: string
  sourcePackage: string
  households: Household[]
  areaName: string
  areaMembers: string[]
  areaOrderTimeS: number
  areaChannel: string
  areaInstruction: string
  areaHazardText: string
  fires: DraftFire[]
}

/**
 * Assemble what the operator drew into the body the backend takes.
 *
 * The alert area is dropped when it has no members, because an area covering nothing is
 * rejected on the far side and the absence of a schedule is a legitimate package.
 */
export function buildDraft(input: DraftInput): PackageDraft {
  const draft: PackageDraft = {
    id: input.id.trim(),
    label: input.label.trim(),
    description: input.description.trim(),
    source_package: input.sourcePackage,
    households: input.households.map((h) => ({ building_id: h.building_id, count: h.count })),
    fires: input.fires,
  }
  if (input.areaMembers.length > 0 && input.areaName.trim()) {
    const name = input.areaName.trim()
    draft.alert_areas = [
      { name, label: name, building_ids: input.areaMembers },
    ]
    draft.alert_events = [
      {
        id: 'EA-1',
        issue_time_s: input.areaOrderTimeS,
        areas: [name],
        instruction: input.areaInstruction,
        channel: input.areaChannel,
        hazard_text: input.areaHazardText.trim(),
      },
    ]
  }
  return draft
}

/** A fire origin with the defaults the record's own sources use. */
export function newFire(index: number, x: number, y: number): DraftFire {
  return {
    id: `source_${index + 1}`,
    x: Math.round(x * 100) / 100,
    y: Math.round(y * 100) / 100,
    t0: 0,
    r0: 30,
    growth_m_per_s: 0.3,
    max_r_m: 700,
  }
}

/**
 * Radius of a fire at a given instant, the same growth the simulator applies.
 *
 * The scrubber uses this to show what the front covers at a chosen moment, so a placed
 * source can be judged against the households it would reach before the package is run.
 */
export function fireRadiusAt(fire: DraftFire, simTimeS: number): number {
  if (simTimeS < fire.t0) return 0
  const grown = fire.r0 + fire.growth_m_per_s * (simTimeS - fire.t0)
  if (fire.max_r_m != null) return Math.max(0, Math.min(grown, fire.max_r_m))
  return Math.max(0, grown)
}

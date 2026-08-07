import type { StyleSpecification } from 'maplibre-gl'
import { COLORS } from '../map/icons'
import { EMPTY_COLLECTION } from '../map/mapStyle'

// The authoring map is the live map's sibling. It draws the same offline basemap from
// the road network, so it needs no tile server and no glyphs, and adds the three layers
// an operator draws against: every building in the package's area, the ones picked as
// households or as the area an order covers, and the fire fronts being placed.
//
// Buildings are drawn as footprints where the bundle carries them, because a footprint
// is a target an operator can hit, and as circles where it does not.

export const AUTHOR_SOURCES = {
  roads: 'roads',
  buildings: 'buildings',
  fires: 'fires',
  box: 'box',
} as const

/** Feature properties the layers read, kept in one place so the view and style agree. */
export const BUILDING_STATE = {
  /** Selected as a household. */
  household: 'hh',
  /** Inside the alert area. */
  area: 'area',
  /** No road within range, so no household can spawn here. */
  stranded: 'stranded',
  /** Holds more than one agent, so it stands out among the ordinary houses. */
  multi: 'multi',
} as const

export function buildAuthorStyle(): StyleSpecification {
  return {
    version: 8,
    name: 'agentevac-authoring',
    sources: {
      [AUTHOR_SOURCES.roads]: { type: 'geojson', data: EMPTY_COLLECTION },
      [AUTHOR_SOURCES.buildings]: { type: 'geojson', data: EMPTY_COLLECTION },
      [AUTHOR_SOURCES.fires]: { type: 'geojson', data: EMPTY_COLLECTION },
      [AUTHOR_SOURCES.box]: { type: 'geojson', data: EMPTY_COLLECTION },
    },
    layers: [
      {
        id: 'canvas',
        type: 'background',
        paint: { 'background-color': '#0B0E13' },
      },
      {
        id: 'roads-context',
        type: 'line',
        source: AUTHOR_SOURCES.roads,
        filter: ['!=', ['get', 'c'], 1],
        paint: {
          'line-color': '#39424E',
          'line-width': ['interpolate', ['linear'], ['zoom'], 8, 0.5, 12, 1.4, 16, 3],
        },
      },
      {
        id: 'roads-core',
        type: 'line',
        source: AUTHOR_SOURCES.roads,
        filter: ['==', ['get', 'c'], 1],
        paint: {
          'line-color': '#4C5665',
          'line-width': ['interpolate', ['linear'], ['zoom'], 8, 0.4, 12, 1.1, 16, 2.6],
        },
      },
      {
        id: 'building-fill',
        type: 'fill',
        source: AUTHOR_SOURCES.buildings,
        paint: {
          'fill-color': [
            'case',
            ['get', BUILDING_STATE.area], COLORS.ordered,
            ['get', BUILDING_STATE.household], COLORS.evacuating,
            ['get', BUILDING_STATE.stranded], '#2A313A',
            '#49535F',
          ],
          'fill-opacity': [
            'case',
            ['get', BUILDING_STATE.area], 0.9,
            ['get', BUILDING_STATE.household], 0.85,
            0.42,
          ],
        },
      },
      // A building holding more than one agent is ringed, so a school or a care home is
      // findable again among several hundred ordinary houses.
      {
        id: 'building-multi',
        type: 'line',
        source: AUTHOR_SOURCES.buildings,
        filter: ['get', BUILDING_STATE.multi],
        paint: {
          'line-color': '#E6EAF0',
          'line-width': 2.2,
          'line-opacity': 0.9,
        },
      },
      // The outline repeats the fill, so the state survives at a zoom where a small
      // footprint is only a few pixels of fill.
      {
        id: 'building-outline',
        type: 'line',
        source: AUTHOR_SOURCES.buildings,
        paint: {
          'line-color': [
            'case',
            ['get', BUILDING_STATE.area], '#FFD08A',
            ['get', BUILDING_STATE.household], '#8FC5E8',
            '#5A6472',
          ],
          'line-width': [
            'case',
            ['get', BUILDING_STATE.area], 2.6,
            ['get', BUILDING_STATE.household], 1.4,
            0.6,
          ],
        },
      },
      // Fire is the only thing on this map allowed to be red, matching the live view.
      {
        id: 'fire-fill',
        type: 'fill',
        source: AUTHOR_SOURCES.fires,
        paint: { 'fill-color': COLORS.fire, 'fill-opacity': 0.2 },
      },
      {
        id: 'fire-outline',
        type: 'line',
        source: AUTHOR_SOURCES.fires,
        paint: {
          'line-color': COLORS.fire,
          'line-width': ['case', ['get', 'selected'], 2.4, 1.4],
        },
      },
      // The box being dragged, drawn over everything so it stays readable.
      {
        id: 'box-fill',
        type: 'fill',
        source: AUTHOR_SOURCES.box,
        paint: { 'fill-color': '#E6EAF0', 'fill-opacity': 0.08 },
      },
      {
        id: 'box-outline',
        type: 'line',
        source: AUTHOR_SOURCES.box,
        paint: { 'line-color': '#E6EAF0', 'line-width': 1.2, 'line-dasharray': [2, 2] },
      },
    ],
  }
}

/** A building footprint, or a small square when the bundle carries centroids only. */
export function buildingFeature(
  building: { id: string; lon: number; lat: number; edge: string | null; poly?: [number, number][] },
  state: { household: boolean; area: boolean; count?: number },
): GeoJSON.Feature {
  const ring: [number, number][] = building.poly?.length
    ? [...building.poly]
    : squareRing(building.lon, building.lat, 8)
  if (ring.length && (ring[0][0] !== ring[ring.length - 1][0] || ring[0][1] !== ring[ring.length - 1][1])) {
    ring.push(ring[0])
  }
  return {
    type: 'Feature',
    id: undefined,
    properties: {
      bid: building.id,
      [BUILDING_STATE.household]: state.household,
      [BUILDING_STATE.area]: state.area,
      [BUILDING_STATE.stranded]: !building.edge,
      [BUILDING_STATE.multi]: (state.count ?? 1) > 1,
      count: state.count ?? 0,
    },
    geometry: { type: 'Polygon', coordinates: [ring] },
  }
}

/** A square of the given half-size in metres, for buildings with no footprint. */
export function squareRing(lon: number, lat: number, halfM: number): [number, number][] {
  const dLat = halfM / 111132
  const dLon = halfM / (111320 * Math.max(0.05, Math.cos((lat * Math.PI) / 180)))
  return [
    [lon - dLon, lat - dLat],
    [lon + dLon, lat - dLat],
    [lon + dLon, lat + dLat],
    [lon - dLon, lat + dLat],
    [lon - dLon, lat - dLat],
  ]
}

/** The rectangle for a drag in progress. */
export function boxFeature(lon1: number, lat1: number, lon2: number, lat2: number): GeoJSON.Feature {
  const minLon = Math.min(lon1, lon2)
  const maxLon = Math.max(lon1, lon2)
  const minLat = Math.min(lat1, lat2)
  const maxLat = Math.max(lat1, lat2)
  return {
    type: 'Feature',
    properties: {},
    geometry: {
      type: 'Polygon',
      coordinates: [[
        [minLon, minLat],
        [maxLon, minLat],
        [maxLon, maxLat],
        [minLon, maxLat],
        [minLon, minLat],
      ]],
    },
  }
}

import type { StyleSpecification } from 'maplibre-gl'
import { COLORS } from './icons'

// The console draws its own basemap from the road geometry the simulator itself
// uses, so the map needs no tile server, no network, and no glyph files. What an
// operator sees is exactly the network the vehicles are driving on.
//
// No layer uses `text-field`, because text in MapLibre needs glyph files served
// over HTTP. Place names are rendered as HTML markers instead, which keeps the
// console working with networking switched off.

export const EMPTY_COLLECTION: GeoJSON.FeatureCollection = { type: 'FeatureCollection', features: [] }

export const SOURCES = {
  roads: 'roads',
  areas: 'areas',
  fires: 'fires',
  households: 'households',
  selection: 'selection',
} as const

export function buildStyle(): StyleSpecification {
  return {
    version: 8,
    name: 'agentevac-offline',
    sources: {
      [SOURCES.roads]: { type: 'geojson', data: EMPTY_COLLECTION },
      [SOURCES.areas]: { type: 'geojson', data: EMPTY_COLLECTION },
      [SOURCES.fires]: { type: 'geojson', data: EMPTY_COLLECTION },
      [SOURCES.households]: { type: 'geojson', data: EMPTY_COLLECTION },
      [SOURCES.selection]: { type: 'geojson', data: EMPTY_COLLECTION },
    },
    layers: [
      {
        id: 'canvas',
        type: 'background',
        paint: { 'background-color': '#0B0E13' },
      },
      // Alert areas sit under the roads so an issued order tints the ground
      // rather than hiding the street network the operator is reading.
      {
        id: 'area-fill',
        type: 'fill',
        source: SOURCES.areas,
        paint: {
          'fill-color': ['case', ['get', 'ordered'], COLORS.ordered, '#3A4550'],
          'fill-opacity': ['case', ['get', 'ordered'], 0.16, 0.06],
        },
      },
      // An area under an order is outlined solid, one still waiting is dashed.
      // Dash patterns cannot be driven by feature data, so the two states are
      // separate layers rather than one expression.
      {
        id: 'area-outline-pending',
        type: 'line',
        source: SOURCES.areas,
        filter: ['!', ['get', 'ordered']],
        paint: {
          'line-color': '#4A5560',
          'line-width': 1,
          'line-dasharray': [3, 3],
        },
      },
      {
        id: 'area-outline-ordered',
        type: 'line',
        source: SOURCES.areas,
        filter: ['get', 'ordered'],
        paint: {
          'line-color': COLORS.ordered,
          'line-width': 2,
        },
      },
      // Two road bands. Local streets carry the incident, arterials carry the
      // corridors out to the shelters.
      {
        id: 'roads-context',
        type: 'line',
        source: SOURCES.roads,
        filter: ['!=', ['get', 'c'], 1],
        paint: {
          'line-color': '#39424E',
          'line-width': ['interpolate', ['linear'], ['zoom'], 8, 0.5, 12, 1.4, 16, 3],
        },
      },
      {
        id: 'roads-core',
        type: 'line',
        source: SOURCES.roads,
        filter: ['==', ['get', 'c'], 1],
        paint: {
          'line-color': '#4C5665',
          'line-width': ['interpolate', ['linear'], ['zoom'], 8, 0.4, 12, 1.1, 16, 2.6],
        },
      },
      // Fire is the only thing on the map allowed to be red.
      {
        id: 'fire-fill',
        type: 'fill',
        source: SOURCES.fires,
        paint: { 'fill-color': COLORS.fire, 'fill-opacity': 0.22 },
      },
      {
        id: 'fire-outline',
        type: 'line',
        source: SOURCES.fires,
        paint: {
          'line-color': COLORS.fire,
          'line-width': 1.6,
          'line-opacity': ['case', ['get', 'fresh'], 1, 0.75],
        },
      },
      {
        id: 'selection-halo',
        type: 'circle',
        source: SOURCES.selection,
        paint: {
          'circle-radius': 13,
          'circle-color': 'rgba(0,0,0,0)',
          'circle-stroke-color': '#E6EAF0',
          'circle-stroke-width': 2,
        },
      },
      {
        id: 'households',
        type: 'symbol',
        source: SOURCES.households,
        layout: {
          'icon-image': ['get', 'icon'],
          'icon-allow-overlap': true,
          'icon-ignore-placement': true,
          'icon-size': ['interpolate', ['linear'], ['zoom'], 9, 0.42, 13, 0.62, 17, 0.9],
        },
      },
    ],
  }
}

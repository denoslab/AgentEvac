import { validateStyleMin } from '@maplibre/maplibre-gl-style-spec'
import { describe, expect, it } from 'vitest'
import { boundsOf, circleRing, distanceM, metresPerDegreeLon, padBounds } from '../map/geo'
import { buildStyle } from '../map/mapStyle'
import { iconFor } from '../map/icons'

describe('map style', () => {
  it('passes the MapLibre style specification', () => {
    // A style error blanks the map at load with nothing on screen to explain it,
    // which is exactly the failure the console must never have on stage.
    expect(validateStyleMin(buildStyle())).toEqual([])
  })

  it('needs no network to render', () => {
    const style = buildStyle()
    expect(style.glyphs).toBeUndefined()
    expect(style.sprite).toBeUndefined()
    for (const source of Object.values(style.sources)) {
      expect(source.type).toBe('geojson')
    }
    // Text layers would need glyph files fetched over HTTP, so place names are
    // drawn as markers instead.
    for (const layer of style.layers) {
      expect((layer as { layout?: Record<string, unknown> }).layout?.['text-field']).toBeUndefined()
    }
  })

  it('keeps hazard red for fire alone', () => {
    const style = buildStyle()
    const red = style.layers.filter((layer) =>
      JSON.stringify((layer as { paint?: unknown }).paint ?? {}).includes('D55E00'),
    )
    expect(red.map((layer) => layer.id).sort()).toEqual(['fire-fill', 'fire-outline'])
  })
})

describe('household markers', () => {
  it('encodes status as a shape, not only a colour', () => {
    expect(iconFor('waiting', false, false)).toBe('hh-waiting')
    expect(iconFor('waiting', true, false)).toBe('hh-waiting-aware')
    expect(iconFor('evacuating', true, false)).toBe('hh-evacuating')
    expect(iconFor('arrived', true, false)).toBe('hh-arrived')
  })

  it('lets fire contact override every other status', () => {
    expect(iconFor('evacuating', true, true)).toBe('hh-contact')
    expect(iconFor('arrived', true, true)).toBe('hh-contact')
  })
})

describe('geodesy', () => {
  it('shrinks a degree of longitude towards the pole', () => {
    expect(metresPerDegreeLon(0)).toBeCloseTo(111320, 0)
    // At the latitude of Upper Tantallon a degree of longitude is about 79 km.
    expect(metresPerDegreeLon(44.72)).toBeCloseTo(79099, -1)
  })

  it('measures ground distance in metres', () => {
    // Roughly one kilometre north at the latitude of Upper Tantallon.
    expect(distanceM(-63.88, 44.72, -63.88, 44.72898)).toBeCloseTo(1000, -1)
  })

  it('draws a fire front as a closed ring of the requested radius', () => {
    const ring = circleRing(-63.88, 44.72, 700)
    expect(ring[0]).toEqual(ring[ring.length - 1])
    expect(ring).toHaveLength(61)
    for (const [lon, lat] of ring) {
      expect(distanceM(-63.88, 44.72, lon, lat)).toBeCloseTo(700, -2)
    }
  })

  it('bounds a point set and pads it without collapsing', () => {
    const bounds = boundsOf([
      { lon: -63.9, lat: 44.7 },
      { lon: -63.8, lat: 44.76 },
    ])
    expect(bounds).toEqual([-63.9, 44.7, -63.8, 44.76])
    const padded = padBounds(bounds!)
    expect(padded[0]).toBeLessThan(bounds![0])
    expect(padded[3]).toBeGreaterThan(bounds![3])

    // A single point still yields a box the map can fit to.
    const single = padBounds(boundsOf([{ lon: -63.88, lat: 44.72 }])!)
    expect(single[2] - single[0]).toBeGreaterThan(0)
  })

  it('has no bounds for an empty set', () => {
    expect(boundsOf([])).toBeNull()
  })
})

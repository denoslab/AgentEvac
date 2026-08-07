import maplibregl, { type GeoJSONSource, type LngLatBoundsLike, type Map as MapLibreMap } from 'maplibre-gl'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useConsole } from '../state/store'
import type { AgentFrame, Preview, Snapshot } from '../state/types'
import { boundsOf, circleRing, distanceM, padBounds, type Bounds } from './geo'
import { iconFor, registerIcons, type IconName } from './icons'
import { buildStyle, EMPTY_COLLECTION, SOURCES } from './mapStyle'

const ATTRIBUTION =
  'Road network from <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a> contributors, built with Eclipse SUMO'

interface Rendered {
  lon: number
  lat: number
  icon: IconName
}

interface Target {
  fromLon: number
  fromLat: number
  toLon: number
  toLat: number
  icon: IconName
}

function householdFeature(id: string, lon: number, lat: number, icon: IconName): GeoJSON.Feature {
  return {
    type: 'Feature',
    id: undefined,
    properties: { id, icon },
    geometry: { type: 'Point', coordinates: [lon, lat] },
  }
}

function areaCollection(preview: Preview | null, snapshot: Snapshot | null): GeoJSON.FeatureCollection {
  if (!preview?.areas?.length) return EMPTY_COLLECTION
  const orderedNames = new Set((snapshot?.areas ?? []).filter((a) => a.ordered).map((a) => a.name))
  return {
    type: 'FeatureCollection',
    features: preview.areas
      .filter((area) => area.hull.length >= 4)
      .map((area) => ({
        type: 'Feature' as const,
        properties: {
          name: area.name,
          ordered: orderedNames.has(area.name),
          order_t_s: area.order_t_s,
        },
        geometry: { type: 'Polygon' as const, coordinates: [area.hull] },
      })),
  }
}

export function LiveMap({ interactive = true }: { interactive?: boolean }) {
  const container = useRef<HTMLDivElement>(null)
  const mapRef = useRef<MapLibreMap | null>(null)
  const [ready, setReady] = useState(false)
  const [zoomTarget, setZoomTarget] = useState<'incident' | 'region'>('incident')

  const preview = useConsole((s) => s.preview)
  const snapshot = useConsole((s) => s.snapshot)
  const session = useConsole((s) => s.session)
  const layers = useConsole((s) => s.layers)
  const selectedAgent = useConsole((s) => s.selectedAgent)
  const selectAgent = useConsole((s) => s.selectAgent)

  const rendered = useRef<Map<string, Rendered>>(new Map())
  const targets = useRef<Map<string, Target>>(new Map())
  const animStart = useRef(0)
  const animDuration = useRef(500)
  const lastSnapshotWall = useRef(0)
  const fireFirstSeen = useRef<Map<string, number>>(new Map())
  const markers = useRef<maplibregl.Marker[]>([])
  const fittedFor = useRef<string | null>(null)

  // ---------------------------------------------------------------- map setup
  useEffect(() => {
    if (!container.current || mapRef.current) return
    const map = new maplibregl.Map({
      container: container.current,
      style: buildStyle(),
      center: [-63.9, 44.73],
      zoom: 10.5,
      attributionControl: false,
      interactive,
      // The map is a picture of a place, not a globe demo.
      pitchWithRotate: false,
      dragRotate: false,
      maxZoom: 18,
    })
    map.addControl(new maplibregl.AttributionControl({ compact: true, customAttribution: ATTRIBUTION }))
    if (interactive) {
      map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right')
    }
    map.on('load', () => {
      registerIcons(map)
      setReady(true)
    })
    if (interactive) {
      map.on('click', 'households', (event) => {
        const id = event.features?.[0]?.properties?.id
        if (typeof id === 'string') selectAgent(id)
      })
      map.on('mouseenter', 'households', () => {
        map.getCanvas().style.cursor = 'pointer'
      })
      map.on('mouseleave', 'households', () => {
        map.getCanvas().style.cursor = ''
      })
    }
    // Folding a panel or dragging a divider changes the map's box without the
    // window changing size, so the canvas is told to follow its container.
    const observer = new ResizeObserver(() => map.resize())
    observer.observe(container.current)

    mapRef.current = map
    return () => {
      observer.disconnect()
      map.remove()
      mapRef.current = null
    }
  }, [interactive, selectAgent])

  // ------------------------------------------------------- static geography
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready || !preview) return
    const roads = map.getSource(SOURCES.roads) as GeoJSONSource | undefined
    roads?.setData(preview.roads ?? EMPTY_COLLECTION)

    for (const marker of markers.current) marker.remove()
    markers.current = []

    for (const destination of preview.destinations ?? []) {
      const element = document.createElement('div')
      element.className =
        'flex -translate-y-1 items-center gap-1 rounded border border-status-nominal/60 bg-ink-bg/90 px-1.5 py-0.5 text-micro font-medium text-status-nominal shadow'
      element.innerHTML =
        '<svg width="9" height="9" viewBox="0 0 10 10" aria-hidden><path d="M5 1 L9 4.5 L7.6 4.5 L7.6 9 L2.4 9 L2.4 4.5 L1 4.5 Z" fill="currentColor"/></svg>' +
        `<span>${destination.name.replace(/_/g, ' ')}</span>`
      markers.current.push(
        new maplibregl.Marker({ element, anchor: 'bottom' })
          .setLngLat([destination.lon, destination.lat])
          .addTo(map),
      )
    }

    for (const area of preview.areas ?? []) {
      if (area.hull.length < 4) continue
      const centre = area.hull.reduce(
        (acc, [lon, lat]) => [acc[0] + lon / area.hull.length, acc[1] + lat / area.hull.length],
        [0, 0],
      )
      const element = document.createElement('div')
      element.className = 'rounded bg-ink-bg/75 px-1.5 py-0.5 text-micro font-medium uppercase tracking-wide text-ink-muted'
      element.textContent = area.name.replace(/_/g, ' ')
      markers.current.push(
        new maplibregl.Marker({ element }).setLngLat([centre[0], centre[1]]).addTo(map),
      )
    }

    const key = preview.package ?? 'live'
    if (fittedFor.current !== key) {
      fittedFor.current = key
      const bounds = (preview.bbox ?? preview.reach_bbox) as Bounds | null
      if (bounds) map.fitBounds(padBounds(bounds) as LngLatBoundsLike, { padding: 40, duration: 0 })
    }
  }, [preview, ready])

  // -------------------------------------------------------------- area state
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const source = map.getSource(SOURCES.areas) as GeoJSONSource | undefined
    source?.setData(areaCollection(preview, snapshot))
  }, [preview, snapshot?.areas, ready])

  // -------------------------------------------------------------- fire fronts
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const source = map.getSource(SOURCES.fires) as GeoJSONSource | undefined
    if (!source) return
    const now = snapshot?.sim_t_s ?? 0
    const fires = snapshot?.fires ?? []
    for (const fire of fires) {
      if (!fireFirstSeen.current.has(fire.id)) fireFirstSeen.current.set(fire.id, now)
    }
    source.setData({
      type: 'FeatureCollection',
      features: fires.map((fire) => ({
        type: 'Feature' as const,
        properties: {
          id: fire.id,
          r_m: fire.r_m,
          // A front that ignited within the last few minutes of incident time
          // reads as new, which is what the ignition treatment marks.
          fresh: now - (fireFirstSeen.current.get(fire.id) ?? now) < 300,
        },
        geometry: {
          type: 'Polygon' as const,
          coordinates: [circleRing(fire.lon, fire.lat, Math.max(20, fire.r_m))],
        },
      })),
    })
  }, [snapshot?.fires, snapshot?.sim_t_s, ready])

  // ------------------------------------------------------- household motion
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return

    // Before a run exists the households sit at home, drawn from the package.
    if (!snapshot) {
      const source = map.getSource(SOURCES.households) as GeoJSONSource | undefined
      rendered.current.clear()
      targets.current.clear()
      source?.setData({
        type: 'FeatureCollection',
        features: (preview?.households ?? []).map((h) =>
          householdFeature(h.id, h.lon, h.lat, 'hh-waiting'),
        ),
      })
      return
    }

    const now = performance.now()
    const gap = lastSnapshotWall.current ? now - lastSnapshotWall.current : 500
    lastSnapshotWall.current = now
    animDuration.current = Math.min(1600, Math.max(160, gap))
    animStart.current = now

    // A household that jumps further than any vehicle could travel has been
    // moved by the simulator rather than driven, so its marker snaps instead of
    // sliding across the map.
    const simGap = Math.max(0.2, snapshot.sim_t_s - (useConsole.getState().previousSnapshot?.sim_t_s ?? snapshot.sim_t_s))
    const snapDistanceM = 45 * simGap + 250

    const next = new Map<string, Target>()
    for (const agent of snapshot.agents as AgentFrame[]) {
      const icon = iconFor(agent.status, agent.aware, Boolean(agent.fire_contact))
      const current = rendered.current.get(agent.id)
      const from = current ?? { lon: agent.lon, lat: agent.lat, icon }
      const jumped = distanceM(from.lon, from.lat, agent.lon, agent.lat) > snapDistanceM
      next.set(agent.id, {
        fromLon: jumped ? agent.lon : from.lon,
        fromLat: jumped ? agent.lat : from.lat,
        toLon: agent.lon,
        toLat: agent.lat,
        icon,
      })
    }
    targets.current = next
  }, [snapshot, preview, ready])

  // The interpolation loop runs outside React so telemetry never triggers a
  // component render at frame rate.
  useEffect(() => {
    if (!ready) return
    let frame = 0
    const step = () => {
      frame = requestAnimationFrame(step)
      const map = mapRef.current
      if (!map || !targets.current.size) return
      const source = map.getSource(SOURCES.households) as GeoJSONSource | undefined
      if (!source) return
      const alpha = Math.min(1, (performance.now() - animStart.current) / animDuration.current)
      const features: GeoJSON.Feature[] = []
      for (const [id, target] of targets.current) {
        const lon = target.fromLon + (target.toLon - target.fromLon) * alpha
        const lat = target.fromLat + (target.toLat - target.fromLat) * alpha
        rendered.current.set(id, { lon, lat, icon: target.icon })
        features.push(householdFeature(id, lon, lat, target.icon))
      }
      source.setData({ type: 'FeatureCollection', features })
    }
    frame = requestAnimationFrame(step)
    return () => cancelAnimationFrame(frame)
  }, [ready])

  // ---------------------------------------------------------------- selection
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const source = map.getSource(SOURCES.selection) as GeoJSONSource | undefined
    if (!source) return
    const position = selectedAgent ? rendered.current.get(selectedAgent) : null
    source.setData(
      position
        ? {
            type: 'FeatureCollection',
            features: [householdFeature(selectedAgent!, position.lon, position.lat, position.icon)],
          }
        : EMPTY_COLLECTION,
    )
  }, [selectedAgent, snapshot, ready])

  // ------------------------------------------------------------ layer toggles
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const mapping: Record<string, string[]> = {
      roads: ['roads-core', 'roads-context'],
      areas: ['area-fill', 'area-outline-pending', 'area-outline-ordered'],
      fires: ['fire-fill', 'fire-outline'],
      households: ['households', 'selection-halo'],
    }
    for (const [key, ids] of Object.entries(mapping)) {
      for (const id of ids) {
        if (map.getLayer(id)) {
          map.setLayoutProperty(id, 'visibility', layers[key] === false ? 'none' : 'visible')
        }
      }
    }
    for (const marker of markers.current) {
      marker.getElement().style.display = layers.destinations === false ? 'none' : ''
    }
  }, [layers, ready, preview])

  // -------------------------------------------------------------- zoom preset
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready || !preview) return
    const bounds = (zoomTarget === 'incident' ? preview.bbox : preview.reach_bbox) as Bounds | null
    if (bounds) map.fitBounds(padBounds(bounds) as LngLatBoundsLike, { padding: 40, duration: 500 })
  }, [zoomTarget, ready, preview])

  const fireBounds = useMemo(() => boundsOf(snapshot?.fires ?? []), [snapshot?.fires])

  const stale = session.phase === 'complete' || session.phase === 'ended_by_operator'

  return (
    <div className="relative h-full w-full overflow-hidden rounded-panel border border-ink-line">
      <div ref={container} className="h-full w-full" />

      {!preview && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <p className="rounded-panel border border-ink-line bg-ink-panel/90 px-4 py-2 text-small text-ink-muted">
            {session.phase === 'preparing'
              ? 'Building the map while the road network loads'
              : 'Select a scenario package to see it on the map'}
          </p>
        </div>
      )}

      {interactive && (
        <div className="absolute left-3 top-3 flex gap-1.5">
          <button
            type="button"
            onClick={() => setZoomTarget('incident')}
            className={`btn h-7 px-2 text-micro ${zoomTarget === 'incident' ? 'border-status-caution text-status-caution' : ''}`}
          >
            Incident area
          </button>
          <button
            type="button"
            onClick={() => setZoomTarget('region')}
            className={`btn h-7 px-2 text-micro ${zoomTarget === 'region' ? 'border-status-caution text-status-caution' : ''}`}
          >
            Shelters in view
          </button>
          {fireBounds && (
            <button
              type="button"
              onClick={() =>
                mapRef.current?.fitBounds(padBounds(fireBounds, 0.4) as LngLatBoundsLike, {
                  padding: 60,
                  duration: 500,
                })
              }
              className="btn h-7 px-2 text-micro"
            >
              Fire front
            </button>
          )}
        </div>
      )}

      {stale && (
        <div className="pointer-events-none absolute right-3 top-3 rounded border border-status-nominal/50 bg-ink-bg/90 px-2 py-1 text-micro font-semibold uppercase tracking-wide text-status-nominal">
          Final position
        </div>
      )}
    </div>
  )
}

import maplibregl, {
  type GeoJSONSource,
  type MapGeoJSONFeature,
  type Map as MapLibreMap,
  type MapMouseEvent,
} from 'maplibre-gl'
import { useEffect, useRef, useState } from 'react'
import { boundsOf, circleRing, padBounds } from '../map/geo'
import { EMPTY_COLLECTION } from '../map/mapStyle'
import type { AuthorBuilding, DraftFire } from '../state/types'
import { AUTHOR_SOURCES, boxFeature, buildAuthorStyle, buildingFeature } from './authorStyle'
import { fireRadiusAt, type Box } from './selection'

export type AuthorTool = 'households' | 'area' | 'fire' | 'count'

export interface AuthorMapProps {
  buildings: AuthorBuilding[]
  roads: GeoJSON.FeatureCollection | null
  /** Buildings selected as households, so the map can fill them. */
  householdIds: Set<string>
  /** Buildings inside the alert area. */
  areaIds: Set<string>
  /** Agents per building, for the ones holding more than one. */
  counts: Map<string, number>
  /** Building whose count stepper is open, if any. */
  anchorId: string | null
  /** Where that stepper should sit, in pixels inside the map box. */
  onAnchorMove: (position: { x: number; y: number } | null) => void
  fires: DraftFire[]
  selectedFire: string | null
  /** Simulation instant the fire fronts are drawn at. */
  previewTimeS: number
  tool: AuthorTool
  /** True while the shift key is held, which makes a drag remove instead of add. */
  subtractive: boolean
  onBox: (box: Box, subtract: boolean) => void
  onBuildingClick: (buildingId: string) => void
  onPlaceFire: (lon: number, lat: number) => void
  onSelectFire: (fireId: string | null) => void
}

/**
 * The map an operator draws a package on.
 *
 * It shares the live map's offline basemap, so it needs no tile server and no glyph
 * files. What it adds is drawing. A drag with the left button pulls a box, and the
 * buildings whose centroid falls inside it are handed back. Dragging the map itself
 * stays on the right button and on the space bar, so the two never fight.
 */
export function AuthorMap(props: AuthorMapProps) {
  const {
    buildings, roads, householdIds, areaIds, counts, anchorId, onAnchorMove,
    fires, selectedFire, previewTimeS, tool, subtractive,
    onBox, onBuildingClick, onPlaceFire, onSelectFire,
  } = props

  const container = useRef<HTMLDivElement>(null)
  const mapRef = useRef<MapLibreMap | null>(null)
  const [ready, setReady] = useState(false)
  const fittedFor = useRef<string | null>(null)

  // Drag state lives in refs, because the map's own handlers fire far faster than
  // React would re-render and the box must follow the cursor without lag.
  const dragStart = useRef<{ lon: number; lat: number } | null>(null)
  const dragMoved = useRef(false)
  const toolRef = useRef(tool)
  const subtractiveRef = useRef(subtractive)
  toolRef.current = tool
  subtractiveRef.current = subtractive

  // ---------------------------------------------------------------- map setup
  useEffect(() => {
    if (!container.current || mapRef.current) return
    const map = new maplibregl.Map({
      container: container.current,
      style: buildAuthorStyle(),
      center: [-63.9, 44.73],
      zoom: 11,
      attributionControl: false,
      pitchWithRotate: false,
      dragRotate: false,
      maxZoom: 19,
    })
    // No attribution control here. It sits where warnings appear and covers them, and
    // the basemap is the simulator's own network, credited in the live view already.
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right')
    // The left button draws a selection box, so it cannot also pan, and shift-drag
    // selects, so it cannot also box-zoom. Both built-in handlers are turned off and
    // panning is re-implemented below on the right and middle buttons.
    map.dragPan.disable()
    map.boxZoom.disable()
    map.on('load', () => setReady(true))

    // Panning with the right button, which the browser would otherwise answer with a
    // context menu instead.
    const canvas = map.getCanvas()
    let panning = false
    let lastX = 0
    let lastY = 0
    const onContextMenu = (event: Event) => event.preventDefault()
    const onPanDown = (event: PointerEvent) => {
      if (event.button !== 2 && event.button !== 1) return
      panning = true
      lastX = event.clientX
      lastY = event.clientY
      canvas.setPointerCapture?.(event.pointerId)
      event.preventDefault()
    }
    const onPanMove = (event: PointerEvent) => {
      if (!panning) return
      map.panBy([lastX - event.clientX, lastY - event.clientY], { duration: 0 })
      lastX = event.clientX
      lastY = event.clientY
    }
    const onPanUp = (event: PointerEvent) => {
      if (!panning) return
      panning = false
      canvas.releasePointerCapture?.(event.pointerId)
    }
    canvas.addEventListener('contextmenu', onContextMenu)
    canvas.addEventListener('pointerdown', onPanDown)
    canvas.addEventListener('pointermove', onPanMove)
    canvas.addEventListener('pointerup', onPanUp)
    canvas.addEventListener('pointercancel', onPanUp)

    const observer = new ResizeObserver(() => map.resize())
    observer.observe(container.current)
    mapRef.current = map
    return () => {
      canvas.removeEventListener('contextmenu', onContextMenu)
      canvas.removeEventListener('pointerdown', onPanDown)
      canvas.removeEventListener('pointermove', onPanMove)
      canvas.removeEventListener('pointerup', onPanUp)
      canvas.removeEventListener('pointercancel', onPanUp)
      observer.disconnect()
      map.remove()
      mapRef.current = null
    }
  }, [])

  // ------------------------------------------------------------ drawing input
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return

    const boxSource = () => map.getSource(AUTHOR_SOURCES.box) as GeoJSONSource | undefined

    const onDown = (event: MapMouseEvent) => {
      if ((event.originalEvent as MouseEvent).button !== 0) return
      if (toolRef.current === 'fire' || toolRef.current === 'count') return
      dragStart.current = { lon: event.lngLat.lng, lat: event.lngLat.lat }
      dragMoved.current = false
    }

    const onMove = (event: MapMouseEvent) => {
      const start = dragStart.current
      if (!start) return
      dragMoved.current = true
      boxSource()?.setData({
        type: 'FeatureCollection',
        features: [boxFeature(start.lon, start.lat, event.lngLat.lng, event.lngLat.lat)],
      })
    }

    const onUp = (event: MapMouseEvent) => {
      const start = dragStart.current
      dragStart.current = null
      boxSource()?.setData(EMPTY_COLLECTION)
      if (!start) return
      if (!dragMoved.current) return // a click, handled by the click listener
      onBox(
        { lon1: start.lon, lat1: start.lat, lon2: event.lngLat.lng, lat2: event.lngLat.lat },
        subtractiveRef.current,
      )
    }

    const onClick = (event: MapMouseEvent) => {
      if (toolRef.current === 'fire') {
        onPlaceFire(event.lngLat.lng, event.lngLat.lat)
        return
      }
      const hit = map.queryRenderedFeatures(event.point, { layers: ['building-fill'] })
      const bid = hit[0]?.properties?.bid
      if (typeof bid === 'string') onBuildingClick(bid)
    }

    const onFireClick = (event: MapMouseEvent & { features?: MapGeoJSONFeature[] }) => {
      const id = event.features?.[0]?.properties?.fid
      if (typeof id === 'string') {
        onSelectFire(id)
        event.preventDefault()
      }
    }

    map.on('mousedown', onDown)
    map.on('mousemove', onMove)
    map.on('mouseup', onUp)
    map.on('click', onClick)
    map.on('click', 'fire-fill', onFireClick)
    return () => {
      map.off('mousedown', onDown)
      map.off('mousemove', onMove)
      map.off('mouseup', onUp)
      map.off('click', onClick)
      map.off('click', 'fire-fill', onFireClick)
    }
  }, [ready, onBox, onBuildingClick, onPlaceFire, onSelectFire])

  // The cursor says what a drag will do before the operator commits to one.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    map.getCanvas().style.cursor = tool === 'fire' ? 'copy' : subtractive ? 'not-allowed' : 'crosshair'
  }, [ready, tool, subtractive])

  // ------------------------------------------------------------------- roads
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    ;(map.getSource(AUTHOR_SOURCES.roads) as GeoJSONSource | undefined)?.setData(roads ?? EMPTY_COLLECTION)
  }, [ready, roads])

  // -------------------------------------------------------------- buildings
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const features = buildings.map((building) =>
      buildingFeature(building, {
        household: householdIds.has(building.id),
        area: areaIds.has(building.id),
        count: counts.get(building.id) ?? 0,
      }),
    )
    ;(map.getSource(AUTHOR_SOURCES.buildings) as GeoJSONSource | undefined)?.setData({
      type: 'FeatureCollection',
      features,
    })
  }, [ready, buildings, householdIds, areaIds, counts])

  // ------------------------------------------------------------------- fires
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const features: GeoJSON.Feature[] = []
    for (const fire of fires) {
      const radius = fireRadiusAt(fire, previewTimeS)
      const centre = fireCentreLonLat(fire, buildings)
      if (!centre) continue
      features.push({
        type: 'Feature',
        properties: { fid: fire.id, selected: fire.id === selectedFire },
        geometry: {
          type: 'Polygon',
          coordinates: [circleRing(centre.lon, centre.lat, Math.max(radius, 12), 48)],
        },
      })
    }
    ;(map.getSource(AUTHOR_SOURCES.fires) as GeoJSONSource | undefined)?.setData({
      type: 'FeatureCollection',
      features,
    })
  }, [ready, fires, selectedFire, previewTimeS, buildings])

  // --------------------------------------------- anchor for the count stepper
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    if (!anchorId) {
      onAnchorMove(null)
      return
    }
    const building = buildings.find((b) => b.id === anchorId)
    if (!building) {
      onAnchorMove(null)
      return
    }
    const report = () => {
      const point = map.project([building.lon, building.lat])
      onAnchorMove({ x: point.x, y: point.y })
    }
    report()
    map.on('move', report)
    return () => {
      map.off('move', report)
    }
  }, [ready, anchorId, buildings, onAnchorMove])

  // ------------------------------------------------------ first fit to the area
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready || buildings.length === 0) return
    const key = `${buildings.length}:${buildings[0]?.id ?? ''}`
    if (fittedFor.current === key) return
    const bounds = boundsOf(buildings)
    if (!bounds) return
    const [west, south, east, north] = padBounds(bounds)
    map.fitBounds([[west, south], [east, north]], { padding: 40, duration: 0 })
    fittedFor.current = key
  }, [ready, buildings])

  return <div ref={container} className="h-full w-full" />
}

/**
 * Longitude and latitude of a fire origin.
 *
 * Fires are held in simulation coordinates, because that is what the package records,
 * and the map needs geography. The nearest building supplies the conversion, since the
 * bundle carries both for every one of them, which keeps the browser free of a
 * projection it has no way to compute.
 */
export function fireCentreLonLat(
  fire: DraftFire,
  buildings: AuthorBuilding[],
): { lon: number; lat: number } | null {
  let best: AuthorBuilding | null = null
  let bestDist = Infinity
  for (const building of buildings) {
    if (building.x == null || building.y == null) continue
    const dx = building.x - fire.x
    const dy = building.y - fire.y
    const dist = dx * dx + dy * dy
    if (dist < bestDist) {
      bestDist = dist
      best = building
    }
  }
  if (!best || best.x == null || best.y == null) return null
  // Offset from that building, converted with the local metres-per-degree scale.
  const dLat = (fire.y - best.y) / 111132
  const dLon = (fire.x - best.x) / (111320 * Math.max(0.05, Math.cos((best.lat * Math.PI) / 180)))
  return { lon: best.lon + dLon, lat: best.lat + dLat }
}

/** Simulation coordinates for a clicked point, inverting {@link fireCentreLonLat}. */
export function lonLatToSim(
  lon: number,
  lat: number,
  buildings: AuthorBuilding[],
): { x: number; y: number } | null {
  let best: AuthorBuilding | null = null
  let bestDist = Infinity
  for (const building of buildings) {
    if (building.x == null || building.y == null) continue
    const dLon = building.lon - lon
    const dLat = building.lat - lat
    const dist = dLon * dLon + dLat * dLat
    if (dist < bestDist) {
      bestDist = dist
      best = building
    }
  }
  if (!best || best.x == null || best.y == null) return null
  const mPerLat = 111132
  const mPerLon = 111320 * Math.max(0.05, Math.cos((best.lat * Math.PI) / 180))
  return {
    x: best.x + (lon - best.lon) * mPerLon,
    y: best.y + (lat - best.lat) * mPerLat,
  }
}

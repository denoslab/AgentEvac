// Small geodesic helpers. The console draws in longitude and latitude, but the
// simulator reasons in metres, so radii and distances cross over here.

const METRES_PER_DEGREE_LAT = 111320

export function metresPerDegreeLon(latitude: number): number {
  return METRES_PER_DEGREE_LAT * Math.cos((latitude * Math.PI) / 180)
}

/** Approximate ground distance between two longitude and latitude pairs, in metres. */
export function distanceM(aLon: number, aLat: number, bLon: number, bLat: number): number {
  const dx = (bLon - aLon) * metresPerDegreeLon((aLat + bLat) / 2)
  const dy = (bLat - aLat) * METRES_PER_DEGREE_LAT
  return Math.hypot(dx, dy)
}

/**
 * A circle of ground radius `radiusM` as a closed ring.
 *
 * Fire fronts are modelled as growing circles in metres, and a map needs them as
 * polygons. Sixty vertices reads as smooth at every zoom the console uses.
 */
export function circleRing(
  lon: number,
  lat: number,
  radiusM: number,
  steps = 60,
): [number, number][] {
  const ring: [number, number][] = []
  const dLat = radiusM / METRES_PER_DEGREE_LAT
  const dLon = radiusM / Math.max(1, metresPerDegreeLon(lat))
  for (let i = 0; i <= steps; i += 1) {
    const angle = (i / steps) * Math.PI * 2
    ring.push([lon + dLon * Math.cos(angle), lat + dLat * Math.sin(angle)])
  }
  return ring
}

export type Bounds = [number, number, number, number]

export function padBounds(bounds: Bounds, factor = 0.08): Bounds {
  const [w, s, e, n] = bounds
  const padX = Math.max(0.002, (e - w) * factor)
  const padY = Math.max(0.002, (n - s) * factor)
  return [w - padX, s - padY, e + padX, n + padY]
}

export function boundsOf(points: { lon: number; lat: number }[]): Bounds | null {
  if (!points.length) return null
  let w = Infinity
  let s = Infinity
  let e = -Infinity
  let n = -Infinity
  for (const point of points) {
    w = Math.min(w, point.lon)
    e = Math.max(e, point.lon)
    s = Math.min(s, point.lat)
    n = Math.max(n, point.lat)
  }
  return [w, s, e, n]
}

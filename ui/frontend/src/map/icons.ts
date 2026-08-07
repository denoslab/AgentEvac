import type { Map as MapLibreMap } from 'maplibre-gl'

// Household markers are drawn as shapes, not just colours, so status survives a
// projector and colour vision deficiency. The shapes are generated in a canvas
// at load time, which keeps the console free of image files to ship or fetch.

export const COLORS = {
  waiting: '#8D99A6',
  waitingAware: '#E69F00',
  evacuating: '#0072B2',
  arrived: '#009E73',
  contact: '#D55E00',
  fire: '#D55E00',
  destination: '#009E73',
  ordered: '#E69F00',
} as const

export type IconName =
  | 'hh-waiting'
  | 'hh-waiting-aware'
  | 'hh-evacuating'
  | 'hh-arrived'
  | 'hh-contact'
  | 'shelter'
  | 'ignition'

const SIZE = 30
const RATIO = 2

type Draw = (ctx: CanvasRenderingContext2D, s: number) => void

function circle(color: string): Draw {
  return (ctx, s) => {
    ctx.beginPath()
    ctx.arc(s / 2, s / 2, s * 0.28, 0, Math.PI * 2)
    ctx.fillStyle = color
    ctx.fill()
    ctx.lineWidth = s * 0.07
    ctx.strokeStyle = 'rgba(8,12,16,0.85)'
    ctx.stroke()
  }
}

function triangle(color: string): Draw {
  return (ctx, s) => {
    const r = s * 0.34
    ctx.beginPath()
    ctx.moveTo(s / 2, s / 2 - r)
    ctx.lineTo(s / 2 + r * 0.88, s / 2 + r * 0.62)
    ctx.lineTo(s / 2 - r * 0.88, s / 2 + r * 0.62)
    ctx.closePath()
    ctx.fillStyle = color
    ctx.fill()
    ctx.lineWidth = s * 0.07
    ctx.strokeStyle = 'rgba(8,12,16,0.85)'
    ctx.stroke()
  }
}

function square(color: string): Draw {
  return (ctx, s) => {
    const r = s * 0.25
    ctx.beginPath()
    ctx.rect(s / 2 - r, s / 2 - r, r * 2, r * 2)
    ctx.fillStyle = color
    ctx.fill()
    ctx.lineWidth = s * 0.07
    ctx.strokeStyle = 'rgba(8,12,16,0.85)'
    ctx.stroke()
  }
}

const flame: Draw = (ctx, s) => {
  const c = s / 2
  const r = s * 0.34
  ctx.beginPath()
  ctx.moveTo(c, c - r)
  ctx.bezierCurveTo(c + r * 0.9, c - r * 0.2, c + r * 0.85, c + r * 0.75, c, c + r)
  ctx.bezierCurveTo(c - r * 0.85, c + r * 0.75, c - r * 0.9, c - r * 0.2, c, c - r)
  ctx.closePath()
  ctx.fillStyle = COLORS.contact
  ctx.fill()
  ctx.lineWidth = s * 0.08
  ctx.strokeStyle = 'rgba(8,12,16,0.9)'
  ctx.stroke()
}

const shelter: Draw = (ctx, s) => {
  const c = s / 2
  const r = s * 0.3
  ctx.beginPath()
  ctx.moveTo(c, c - r)
  ctx.lineTo(c + r, c)
  ctx.lineTo(c + r * 0.62, c)
  ctx.lineTo(c + r * 0.62, c + r * 0.75)
  ctx.lineTo(c - r * 0.62, c + r * 0.75)
  ctx.lineTo(c - r * 0.62, c)
  ctx.lineTo(c - r, c)
  ctx.closePath()
  ctx.fillStyle = COLORS.destination
  ctx.fill()
  ctx.lineWidth = s * 0.07
  ctx.strokeStyle = 'rgba(8,12,16,0.9)'
  ctx.stroke()
}

const ignition: Draw = (ctx, s) => {
  const c = s / 2
  ctx.beginPath()
  ctx.arc(c, c, s * 0.3, 0, Math.PI * 2)
  ctx.lineWidth = s * 0.1
  ctx.strokeStyle = COLORS.fire
  ctx.stroke()
  ctx.beginPath()
  ctx.arc(c, c, s * 0.1, 0, Math.PI * 2)
  ctx.fillStyle = COLORS.fire
  ctx.fill()
}

const DRAWINGS: Record<IconName, Draw> = {
  'hh-waiting': circle(COLORS.waiting),
  'hh-waiting-aware': circle(COLORS.waitingAware),
  'hh-evacuating': triangle(COLORS.evacuating),
  'hh-arrived': square(COLORS.arrived),
  'hh-contact': flame,
  shelter,
  ignition,
}

/** Draw every marker once and hand it to the map. Safe to call repeatedly. */
export function registerIcons(map: MapLibreMap): void {
  for (const [name, draw] of Object.entries(DRAWINGS) as [IconName, Draw][]) {
    if (map.hasImage(name)) continue
    const canvas = document.createElement('canvas')
    canvas.width = SIZE * RATIO
    canvas.height = SIZE * RATIO
    const ctx = canvas.getContext('2d')
    if (!ctx) continue
    ctx.scale(RATIO, RATIO)
    draw(ctx, SIZE)
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height)
    map.addImage(name, data, { pixelRatio: RATIO })
  }
}

export function iconFor(status: string, aware: boolean, fireContact: boolean): IconName {
  if (fireContact) return 'hh-contact'
  if (status === 'evacuating') return 'hh-evacuating'
  if (status === 'arrived') return 'hh-arrived'
  return aware ? 'hh-waiting-aware' : 'hh-waiting'
}

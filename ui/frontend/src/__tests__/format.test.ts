import { describe, expect, it } from 'vitest'
import { anchoredClock, duration, integer, percent, simClock, speedLabel } from '../state/format'

// Every readout an operator reads under pressure comes through these functions,
// so the edge cases matter more than the happy path.

describe('simClock', () => {
  it('reads as hours, minutes, and seconds into the incident', () => {
    expect(simClock(0)).toBe('0:00:00')
    expect(simClock(6300)).toBe('1:45:00')
    expect(simClock(28800)).toBe('8:00:00')
  })

  it('says nothing rather than something wrong when there is no time', () => {
    expect(simClock(null)).toBe('--:--:--')
    expect(simClock(undefined)).toBe('--:--:--')
    expect(simClock(NaN)).toBe('--:--:--')
  })
})

describe('anchoredClock', () => {
  it('places simulation time on the local clock of the real incident', () => {
    // The Tantallon reconstruction anchors second zero at 15:28 on 28 May 2023.
    expect(anchoredClock(0, '15:28:00')).toBe('15:28')
    expect(anchoredClock(6300, '15:28:00')).toBe('17:13')
    expect(anchoredClock(15180, '15:28:00')).toBe('19:41')
  })

  it('wraps past midnight instead of running to 25 o clock', () => {
    expect(anchoredClock(32400, '15:28:00')).toBe('00:28')
  })

  it('returns nothing when the package carries no anchor', () => {
    expect(anchoredClock(600, null)).toBeNull()
    expect(anchoredClock(600, 'not a time')).toBeNull()
  })
})

describe('duration', () => {
  it('changes unit as the span grows', () => {
    expect(duration(45)).toBe('45 s')
    expect(duration(900)).toBe('15 min')
    expect(duration(3600)).toBe('1 h')
    expect(duration(28800)).toBe('8 h')
    expect(duration(5400)).toBe('1 h 30 min')
  })

  it('keeps the sign on a negative margin, which means the fire arrived first', () => {
    expect(duration(-45)).toBe('-45 s')
    expect(duration(-295)).toBe('-4 min')
    expect(duration(-900)).toBe('-15 min')
  })
})

describe('numbers', () => {
  it('formats counts and shares', () => {
    expect(integer(1234)).toBe('1,234')
    expect(integer(null)).toBe('--')
    expect(percent(0.6703)).toBe('67%')
    expect(percent(null)).toBe('--')
  })

  it('names the unthrottled speed rather than showing zero', () => {
    expect(speedLabel(60)).toBe('60x')
    expect(speedLabel(0)).toBe('max')
    expect(speedLabel(null)).toBe('--')
  })
})

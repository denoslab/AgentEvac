import { useConsole } from './store'
import type { StreamMessage } from './types'

// The browser's EventSource reconnects on its own, but it says nothing about
// when. This wrapper reports the wait so the connection badge can count down
// rather than leaving an operator staring at a stalled screen.

const RETRY_STEPS_S = [1, 2, 5, 10, 15]

let source: EventSource | null = null
let attempt = 0
let retryTimer: number | null = null
let countdownTimer: number | null = null

function clearTimers() {
  if (retryTimer !== null) window.clearTimeout(retryTimer)
  if (countdownTimer !== null) window.clearInterval(countdownTimer)
  retryTimer = null
  countdownTimer = null
}

function scheduleRetry() {
  const wait = RETRY_STEPS_S[Math.min(attempt, RETRY_STEPS_S.length - 1)]
  attempt += 1
  let remaining = wait
  useConsole.getState().setConnection('retrying', remaining)
  countdownTimer = window.setInterval(() => {
    remaining -= 1
    useConsole.getState().setConnection('retrying', Math.max(0, remaining))
  }, 1000)
  retryTimer = window.setTimeout(() => {
    clearTimers()
    connectStream()
  }, wait * 1000)
}

export function connectStream() {
  clearTimers()
  if (source) {
    source.close()
    source = null
  }
  useConsole.getState().setConnection('connecting')
  const next = new EventSource('/api/stream')
  source = next

  next.onopen = () => {
    attempt = 0
    useConsole.getState().setConnection('open')
  }

  next.onmessage = (event) => {
    let message: StreamMessage
    try {
      message = JSON.parse(event.data)
    } catch {
      return
    }
    useConsole.getState().ingest(message)
  }

  next.onerror = () => {
    // EventSource fires this for both a dropped connection and a failed connect.
    // Either way the wrapper owns the retry so the countdown stays truthful.
    next.close()
    if (source === next) source = null
    scheduleRetry()
  }
}

export function disconnectStream() {
  clearTimers()
  if (source) {
    source.close()
    source = null
  }
  useConsole.getState().setConnection('closed')
}

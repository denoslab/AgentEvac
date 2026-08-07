import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './app/App'
// Vendor stylesheets first, so the console's own rules win where they overlap.
import 'maplibre-gl/dist/maplibre-gl.css'
import 'uplot/dist/uPlot.min.css'
import './index.css'

const container = document.getElementById('root')
if (!container) throw new Error('the console needs a #root element to mount into')

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Dark neutral chrome so the map stays the brightest object on a projector.
        ink: {
          bg: '#12161C',
          panel: '#1A2028',
          raised: '#232B35',
          line: '#2A333E',
          text: '#E6EAF0',
          muted: '#9AA5B1',
          faint: '#6C7681',
        },
        // Status hues from the Okabe-Ito palette, which stay distinguishable
        // under every common form of colour vision deficiency.
        status: {
          nominal: '#009E73',
          caution: '#E69F00',
          hazard: '#D55E00',
          moving: '#0072B2',
          idle: '#8D99A6',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      fontSize: {
        micro: ['12px', '16px'],
        small: ['13px', '18px'],
        base: ['14px', '20px'],
        panel: ['16px', '22px'],
        view: ['20px', '28px'],
        readout: ['28px', '32px'],
      },
      borderRadius: { panel: '6px' },
    },
  },
  plugins: [],
}

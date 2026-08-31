/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        gov: {
          /* Saffron — primary government portal theme */
          saffron:        '#E87516',
          'saffron-dark': '#C85D00',
          'saffron-mid':  '#F28C28',
          'saffron-light':'#FFF1DD',
          cream:          '#FFF7EA',
          /* Structural */
          border:  '#E2E2E2',
          canvas:  '#FFFFFF',
          card:    '#FFFFFF',
          text:    '#172033',
          muted:   '#64748B',
          /* Legacy aliases (kept for backward-compat, now point to saffron) */
          blue:         '#E87516',
          'blue-dark':  '#C85D00',
          'blue-light': '#FFF1DD',
          'blue-hover': '#F28C28',
          header:       '#172033',
        },
        risk: {
          low:      '#16A34A',
          moderate: '#EAB308',
          high:     '#EA580C',
          veryhigh: '#DC2626',
          extreme:  '#7E22CE',
          blue:     '#0284C7',
        }
      },
      fontFamily: {
        sans: [
          'Noto Sans',
          'Inter',
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'Helvetica',
          'Arial',
          'sans-serif',
        ],
      },
      boxShadow: {
        'gov':    '0 1px 3px rgba(0,0,0,0.05)',
        'gov-sm': '0 1px 2px rgba(0,0,0,0.04)',
      }
    },
  },
  plugins: [],
}

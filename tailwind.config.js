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
          blue: '#0A3871',
          'blue-dark': '#06254F',
          'blue-light': '#EBF3FC',
          'blue-hover': '#134D96',
          header: '#002B49',
          border: '#E2E8F0',
          canvas: '#F4F6F9',
          card: '#FFFFFF',
          text: '#0F172A',
          muted: '#64748B',
        },
        risk: {
          low: '#16A34A',
          moderate: '#EAB308',
          high: '#EA580C',
          veryhigh: '#DC2626',
          extreme: '#7E22CE',
          blue: '#0284C7',
        }
      },
      fontFamily: {
        sans: ['Inter', 'Roboto', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Helvetica', 'Arial', 'sans-serif'],
      },
      boxShadow: {
        'gov': '0 1px 3px rgba(0, 0, 0, 0.05)',
        'gov-sm': '0 1px 2px rgba(0, 0, 0, 0.04)',
      }
    },
  },
  plugins: [],
}

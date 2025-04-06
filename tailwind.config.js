/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    './pages/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './app/**/*.{ts,tsx}',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        cosmic: {
          900: '#1a0b2e',
          800: '#2b1a4d',
          700: '#3c296b',
          600: '#4d3889',
          500: '#5e47a7',
        },
        nebula: {
          500: '#8b5cf6',
          600: '#7c3aed',
        },
        glass: 'rgba(255, 255, 255, 0.1)',
      },
      animation: {
        pulseGlow: 'pulseGlow 2s infinite',
        float: 'float 3s ease-in-out infinite',
      },
      keyframes: {
        pulseGlow: {
          '0%, 100%': { boxShadow: '0 0 10px rgba(139, 92, 246, 0.5)' },
          '50%': { boxShadow: '0 0 20px rgba(139, 92, 246, 0.8)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
}
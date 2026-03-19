import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        night: {
          950: "#040712",
          900: "#0a1021",
          800: "#111b35",
        },
        aurora: "#61dafb",
        gold: "#f7b955",
        coral: "#ff7f6e",
      },
      fontFamily: {
        sans: ["Manrope", "Avenir Next", "Segoe UI", "sans-serif"],
        display: ["Space Grotesk", "Avenir Next", "sans-serif"],
      },
      boxShadow: {
        panel: "0 20px 80px rgba(3, 7, 18, 0.35)",
      },
      backgroundImage: {
        grid: "radial-gradient(circle at top, rgba(97, 218, 251, 0.16), transparent 35%), linear-gradient(rgba(255,255,255,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.06) 1px, transparent 1px)",
      },
    },
  },
  plugins: [],
} satisfies Config;

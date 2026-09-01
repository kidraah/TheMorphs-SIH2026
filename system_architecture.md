# VARUNA System Architecture & Technology Stack

This document provides a comprehensive breakdown of the VARUNA (AI-Driven Hyper-Local Early Warning System) architecture from scratch, detailing the technologies used across the frontend and backend, why they were chosen, and the specific benefits they provide to the project.

---

## 1. Frontend Architecture

The frontend is built to be a high-performance, interactive, and visually stunning dashboard capable of rendering complex 3D geospatial data in real-time.

### Core Framework: React + Vite
* **What it is:** React is a component-based UI library, and Vite is a next-generation build tool.
* **Why it's used:** We needed a modular architecture where complex UI components (like the Risk Map, Timeline, and Alert panels) could be developed and maintained independently. Vite was chosen over Create React App or Webpack for its near-instant cold server start times and lightning-fast Hot Module Replacement (HMR).
* **Benefits:** 
  * Extremely fast development experience.
  * Highly optimized production builds.
  * Reusable component architecture keeps the codebase clean.

### State Management: Zustand
* **What it is:** A small, fast, and scalable bear-bones state management solution.
* **Why it's used:** Managing global state (selected timeframes, selected regions, active map layers, and fetched dashboard data) across deeply nested components can become messy with React Context or overly boilerplate-heavy with Redux.
* **Benefits:**
  * Zero boilerplate; hooks directly into components.
  * Prevents unnecessary re-renders, which is critical for performance when the app also has to drive a heavy 3D WebGL canvas.

### Geospatial Rendering: Deck.gl & MapLibre GL JS
* **What it is:** Deck.gl is a WebGL-powered framework for visual exploratory data analysis of large datasets. MapLibre GL JS is an open-source map rendering library.
* **Why it's used:** Rendering 100,000+ data points (like high-resolution flood trajectories or district risk polygons) on a standard Leaflet map would crash the browser. Deck.gl utilizes the GPU to render massive amounts of spatial data over a 3D terrain base map provided by MapLibre.
* **Benefits:**
  * **GPU Acceleration:** Can handle millions of data points smoothly at 60 FPS.
  * **3D Terrain:** Allows us to extrude the Himalayan terrain (AWS DEM) so meteorologists can visually see valleys and slopes where flood risks aggregate.
  * **Layer Composition:** Easily compose heatmaps, GeoJSON polygons, and animated radar tracks on top of each other.

### Styling: Tailwind CSS
* **What it is:** A utility-first CSS framework.
* **Why it's used:** To quickly build out the custom, premium, and government-grade aesthetic required for the dashboard without writing thousands of lines of custom CSS files.
* **Benefits:**
  * Rapid UI iteration.
  * Highly consistent design system (colors, spacing, typography are all standardized).
  * Very small production CSS bundle sizes (it purges unused classes).

### Data Visualization: Recharts
* **What it is:** A composable charting library built on React components.
* **Why it's used:** Used to render the Risk Timeline bell-curve chart.
* **Benefits:** Easily handles responsive SVG charts and integrates seamlessly with React's component lifecycle.

---

## 2. Backend Architecture

The backend is engineered to handle heavy machine learning inference and complex geospatial matrix transformations rapidly, acting as the bridge between the AI model and the frontend dashboard.

### Core Framework: FastAPI (Python)
* **What it is:** A modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.
* **Why it's used:** Python is mandatory because the AI model is built in PyTorch. FastAPI was chosen over Django or Flask because of its native asynchronous (`async`/`await`) support and unparalleled speed (comparable to NodeJS and Go).
* **Benefits:**
  * **High Concurrency:** Can handle multiple rapid requests from the dashboard without blocking the main thread.
  * **Automatic Documentation:** Auto-generates OpenAPI (Swagger) documentation.
  * **Type Safety:** Catches bugs before runtime.

### AI Inference Engine: PyTorch
* **What it is:** The premier open-source machine learning framework.
* **Why it's used:** The core VARUNA model (Hybrid TransU-Net) is built, trained, and executed using PyTorch. 
* **Benefits:**
  * Handles multi-modal tensor operations efficiently.
  * Native support for GPU acceleration (CUDA) if deployed on specialized hardware.

### Geospatial & Array Processing: NumPy, SciPy, Rasterio, & GeoPandas
* **What it is:** The standard suite of Python scientific computing and spatial libraries.
* **Why it's used:** The AI model outputs raw 2D probability matrices. These libraries are used to translate those matrices into real-world geographic coordinates, run Zonal Statistics against district shapefiles, and simulate hydrological flow (D8 routing).
* **Benefits:**
  * **NumPy/SciPy:** Lightning-fast matrix multiplication and Gaussian filtering for radar image generation.
  * **GeoPandas & Rasterio:** Allows us to instantly map the AI's 1km pixel predictions to specific districts (like Uttarkashi or Shimla) and generate the district risk rankings.

### Data Communication & Rendering: Base64 & Matplotlib
* **What it is:** Techniques for transmitting image data over standard JSON APIs.
* **Why it's used:** For the Heatmap and Precipitation Nowcast radar echo, instead of saving PNGs to a slow disk and serving them statically, the backend uses Matplotlib in-memory to generate the heatmaps, encodes them in Base64, and ships them directly in the JSON response payload.
* **Benefits:**
  * **Statelessness:** The API remains entirely stateless, making it infinitely horizontally scalable.
  * **Speed:** No disk I/O bottleneck.

---

*Note: This document covers the current end-to-end implementation.*

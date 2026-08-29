# CartoDEM in Flash Flood & Cloudburst Modeling

## 1. Hydrological Routing
In the SIH26077 pipeline, predicting that 150mm of rain will fall (via INSAT/IMDAA) is only step one. CartoDEM is required for step two: figuring out where that water goes. 
- **Flow Accumulation:** Calculating how much water drains into a specific pixel.
- **Catchment Delineation:** Mapping the exact boundaries of a river valley (e.g., the Mandakini or Bhagirathi rivers).

## 2. Terrain Influence on Convection (Orographic Lift)
CartoDEM is fed directly into the Deep Learning model. The `(Lat, Lon)` grid of the DEM provides the `elevation` and `slope` features. The neural network learns that if moist wind (from IMDAA) is blowing perpendicular to a steep slope (from CartoDEM), forced ascent will occur, dramatically increasing the probability of a cloudburst.

## 3. Inundation Mapping
Once a flash flood is triggered, CartoDEM is used in hydrodynamic models (like HEC-RAS or simplified cellular automata) to simulate water depth and predict which villages and roads will be submerged.

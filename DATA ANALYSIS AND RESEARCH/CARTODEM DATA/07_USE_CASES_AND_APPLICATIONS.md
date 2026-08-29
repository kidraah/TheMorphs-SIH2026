# Use Cases and Applications of CartoDEM

## 1. Flash Flood Routing (SIH26077)
Once the AI predicts a cloudburst over a specific grid using IMDAA, we feed that rainfall volume into a hydrodynamic model layered on top of CartoDEM. The DEM dictates exactly which valleys will flood, how fast the water will travel downstream, and which villages will be submerged.

## 2. Catchment Delineation for Dams
Engineers use CartoDEM to determine the exact boundaries of a river basin. If a dam is built in Tehri, CartoDEM can calculate exactly how many square kilometers of rainfall will drain into that specific dam.

## 3. Landslide Susceptibility Mapping
By analyzing the `Slope` (steepness) and `Aspect` (sun exposure, which affects soil moisture) derived from CartoDEM, geologists map out slopes that are mathematically prone to catastrophic failure during heavy rain.

## 4. Telecom and Defense
Used to calculate "Line of Sight" for placing 5G towers or military radar, ensuring mountains do not block the signal.

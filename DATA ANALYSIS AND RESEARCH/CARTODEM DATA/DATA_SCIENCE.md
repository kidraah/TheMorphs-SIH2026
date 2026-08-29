# The Science of CartoDEM: Geomorphology & Hydrology

## 1. Flow Direction and the D8 Algorithm
The core of using a DEM for flood prediction relies on the **D8 algorithm**. 
For any given pixel, the algorithm checks the elevation of its 8 neighboring pixels. Water is assumed to flow in the direction of the steepest downward slope. This simple physical rule allows the computer to construct a massive web of river networks across the Himalayas purely from CartoDEM.

## 2. Stream Power Index (SPI) & Topographic Wetness Index (TWI)
From the DEM, we derive complex scientific indices:
- **TWI:** Measures the tendency of water to accumulate in a specific area based on slope and catchment size. High TWI areas are prime flash flood zones.
- **SPI:** Measures the erosive power of flowing water. In a cloudburst, high SPI indicates areas where massive landslides and debris flows are likely to be triggered by the deluge.

## 3. The "Funnel" Effect in the Himalayas
Cloudbursts are particularly deadly in "V-shaped" Himalayan valleys. CartoDEM mathematically quantifies this valley geometry. When extreme rain (from the cloudburst) hits the vast upper catchment, the DEM proves that all that volume is forced into a narrow gorge at the bottom, explaining why a 100mm rain event translates to a 15-meter wall of water downstream.

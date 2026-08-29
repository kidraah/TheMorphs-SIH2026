# What Does CartoDEM Show?

## 1. The Grid
CartoDEM divides the landmass of India into a massive grid of pixels. In the publicly available version, each pixel represents a **30m x 30m** square of land.

## 2. The Z-Value (Elevation)
For every single 30m pixel, CartoDEM stores exactly one number: **Elevation in meters above Mean Sea Level (MSL).**
- If a pixel covers the peak of Nanda Devi, the value is ~7,816.
- If a pixel covers the Ganges plains in Haridwar, the value is ~300.

## 3. Derived Variables
While the raw file only shows height, GIS software instantly calculates:
- **Slope:** How steep the ground is (in degrees).
- **Aspect:** Which direction the slope faces (North, South, etc.).
- **Curvature:** Whether the slope is convex (a ridge) or concave (a valley).

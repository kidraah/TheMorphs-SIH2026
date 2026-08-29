# CartoDEM Data Architecture

## 1. Formats
CartoDEM data is distributed primarily in **GeoTIFF** format.
- A GeoTIFF is a standard TIFF image file with embedded geospatial metadata (coordinate reference system, bounding box).
- Each pixel in the raster image represents the elevation (in meters) above mean sea level.

## 2. Tiles and Grids
The data is disseminated in distinct tiles (e.g., 1° x 1° grid squares). For a project covering Uttarakhand and Himachal Pradesh, multiple adjacent tiles must be downloaded from Bhuvan and mosaicked together using GIS software (like QGIS) or Python libraries (like `rasterio`).

## 3. Hydrologically Conditioned DEM
Raw DEMs often contain "sinks" or "pits"—artificial depressions caused by sensor errors. Before CartoDEM can be used in flash flood prediction, it must undergo **Pit Filling**. A hydrologically conditioned DEM ensures that simulated water can flow continuously across the terrain without getting trapped in artificial holes.

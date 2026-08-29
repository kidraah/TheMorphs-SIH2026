# How to Read and Identify CartoDEM Data

## 1. The Format
CartoDEM is primarily distributed as **GeoTIFF** files. 
A GeoTIFF looks like a regular grayscale image if you open it in a basic photo viewer (where white is high mountains and black is sea level). However, it contains embedded georeferencing tags that map every pixel to a precise Latitude and Longitude.

## 2. How to Read It
- **GIS Software:** You cannot analyze a DEM properly with standard photo software. You must use **QGIS** (open-source) or ArcGIS.
- **Python Code:** 
  - `rasterio`: The standard library for reading and writing GeoTIFF arrays.
  - `gdal`: The Geospatial Data Abstraction Library, the underlying C++ engine for almost all geospatial operations.
  - `pysheds`: A Python library specifically built to run hydrological routing (simulating rivers) over DEM arrays.

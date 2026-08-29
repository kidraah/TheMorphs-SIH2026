# IMDAA Data Types and Architecture

## 1. File Formats
IMDAA data is primarily disseminated in **GRIB2** and **NetCDF4** formats, which are standard in atmospheric sciences for handling n-dimensional grid data.

## 2. Vertical Structure
The model utilizes a hybrid height vertical coordinate system with **upper-air variables provided at 37 standard pressure levels** ranging from 1000 hPa (near surface) to 1 hPa (stratosphere).

## 3. Key Variables for Cloudburst Nowcasting
To predict extreme convection, specific variables are extracted from the IMDAA grid:
- **Surface Level / Single Level:**
  - `CAPE` (Convective Available Potential Energy): Joules/kg.
  - `CIN` (Convective Inhibition): Joules/kg.
  - `IWV` (Integrated Water Vapor / Precipitable Water): kg/m² or mm.
  - `Surface Pressure` and `2m Temperature`.
- **Pressure Levels (850hPa, 700hPa, 500hPa, 200hPa):**
  - `U-wind` and `V-wind`: Used to calculate Vertical Wind Shear.
  - `Specific Humidity (q)`: Crucial at 850hPa for low-level moisture transport.
  - `Geopotential Height (Z)`: To track troughs and low-pressure systems.
  - `Vertical Velocity (Omega)`: Indicates the strength of updrafts.

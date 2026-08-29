# INSAT Data Types and Access

## 1. Formats and Distribution
Data is disseminated by the Meteorological & Oceanographic Satellite Data Archival Centre (MOSDAC) in **HDF5 (Hierarchical Data Format)**. HDF5 is optimized for large, complex, multi-dimensional array data.

## 2. L1B and L2B Products
- **L1B (Level 1B):** Raw radiances and brightness temperatures directly from the sensor, georeferenced.
- **L2B (Level 2B):** Scientifically derived geophysical parameters. Key products include:
  - `CTT` (Cloud Top Temperature).
  - `HEM` (Hydro Estimator Method) Quantitative Precipitation Estimation (QPE).
  - `AMV` (Atmospheric Motion Vectors).
  - `OLR` (Outgoing Longwave Radiation).

## 3. Metadata and Quality Flags
INSAT HDF5 files contain extensive metadata, including sensor zenith angles, solar zenith angles, and quality flags (to mask out corrupted scan lines or eclipse periods).

# How to Read and Identify IMDAA Data

## 1. File Formats
IMDAA is not a standard spreadsheet or image. It is packaged in scientific binary formats:
- **GRIB2 (GRIdded Binary):** The WMO standard. Highly compressed, designed for meteorology.
- **NetCDF (Network Common Data Form):** An array-oriented format that includes self-describing metadata.

## 2. Identifying the Data
When you open an IMDAA file, you must identify variables by their scientific "short names":
- `t2m` = Temperature at 2 meters.
- `u10` / `v10` = U/V wind vectors at 10 meters.
- `cape` = Convective Available Potential Energy.

## 3. How to Read It (Software & Code)
Because the data is a massive 4D tensor, standard software cannot open it.
- **Visual Tools:** `Panoply` (by NASA) or `ncview`.
- **Python Libraries:** 
  - `xarray`: The industry standard for loading NetCDF files as multi-dimensional arrays (tensors).
  - `cfgrib`: An engine for `xarray` to decode GRIB2 files.
  - `MetPy`: Used to calculate complex derivatives (like wind shear) from the raw arrays.

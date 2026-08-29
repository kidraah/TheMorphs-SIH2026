# How to Read and Identify INSAT Data

## 1. The Format
INSAT data is distributed in **HDF5** (Hierarchical Data Format, v5). It acts like a digital filing cabinet, containing multi-dimensional arrays (the images) alongside extensive metadata (calibration constants, angles).

## 2. Digital Numbers vs. Brightness Temperature
When you read an HDF5 file, the raw array is filled with "Digital Numbers" (DN) ranging from 0 to 1023 (10-bit). 
To make it useful, you must apply a calibration formula (found in the metadata) to convert the DN into **Radiance**, and then use the Inverse Planck Function to convert Radiance into **Brightness Temperature (Kelvin)**.

## 3. Tools to Read It
- **Software:** HDFView, QGIS (with HDF plugins).
- **Code:** Python using the `h5py` or `netCDF4` libraries. Data scientists load the HDF5 arrays into `numpy` matrices, apply the calibration math, and plot it using `matplotlib` or `Cartopy`.

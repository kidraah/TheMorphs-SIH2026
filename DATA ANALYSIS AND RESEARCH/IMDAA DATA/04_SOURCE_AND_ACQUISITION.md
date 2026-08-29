# Source and Acquisition of IMDAA

## 1. Who Produces It?
IMDAA is produced by the **National Centre for Medium Range Weather Forecasting (NCMRWF)** under the Ministry of Earth Sciences (MoES), India, in collaboration with the UK Met Office.

## 2. How the Government Gets/Creates the Data
The data doesn't come from a single sensor. The NCMRWF supercomputers (like *Mihir* and *Pratyush*) ingest petabytes of historical raw data:
- **Satellites:** ISRO (INSAT) and global satellites (NOAA, METEOSAT).
- **Ground Stations:** IMD rain gauges, automated weather stations (AWS).
- **Aviation:** Commercial aircraft sensors (AMDAR).
- **Marine:** Ocean buoys and ship logs.

The supercomputer runs the **Unified Model (UM)** (a complex physics simulation). Through a process called **Data Assimilation**, the raw observations are mathematically blended into the UM's grid, forcing the physics simulation to match reality. The output of this simulation is the IMDAA dataset.

## 3. How We Access It
Researchers and developers access it via the **NCMRWF Data Portal** or the **RDA (Research Data Archive)** by establishing FTP/API connections to download specific variables and timeframes.

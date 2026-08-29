# Source and Acquisition of INSAT Data

## 1. From Space to Earth
- **Satellites:** Designed and launched by ISRO (Indian Space Research Organisation). 
- **Payloads:** The Imager and Sounder instruments are optical/infrared telescopes built by the Space Applications Centre (SAC), Ahmedabad.
- **Downlink:** As the satellite scans the earth, it beams the raw telemetry down in real-time to the Master Control Facility (MCF) in Hassan, Karnataka, and the IMD data receiving center in New Delhi.

## 2. Processing and Distribution
The raw telemetry is useless to standard users. It is passed to **MOSDAC** (Meteorological and Oceanographic Satellite Data Archival Centre), managed by ISRO.
MOSDAC supercomputers geographically register the pixels (stretching the image so it maps perfectly to Earth's coordinates) and apply calibration.

## 3. How We Get It
Researchers, government bodies (like NDMA), and developers download the L1B and L2B HDF5 products directly from the **MOSDAC web portal** via automated FTP/HTTP scripts in near-real-time.

"""The frozen cache config for INSAT bulk ingest. Enumerated, not remembered.

Why this file exists as code rather than a checklist
----------------------------------------------------
Bulk ingest transfers ~3.4 TB and decodes it once. Anything omitted here is
not a small fix later -- it is a re-download of 3.4 TB, because MOSDAC does
not subset server-side (measured: 702 delivered files are full Earth disk,
Latitude -81.04..+81.04, Longitude -7.15..+155.15) and the raw files are
discarded after decoding.

So every decision that changes the bytes on disk is a FIELD, the fingerprint
covers all of them, and a cache built with one config refuses to load under
another. A checklist in a document cannot do that.

THE CHANNEL DECISION, AND WHY IT IS "ALL OF THEM"
-------------------------------------------------
Measured stored bytes inside one real 448 MB scan:

    Longitude_VIS  139.1 MB  30.8%      IMG_VIS   80.5 MB  17.8%
    Latitude_VIS    87.9 MB  19.5%      other     57.3 MB  12.7%
    IMG_SWIR        86.4 MB  19.2%

The two channels the model currently reads are 19.1 MB -- 4.2% of the file.
Every VIS and SWIR byte is transferred whether or not it is kept. On the
4 km analysis grid a channel costs ~1.6 MB per scan, so keeping all six
costs ~47 GB more storage across the whole archive and removes any
possibility of a 3.4 TB re-download to recover one.

Asymmetric cost, so: extract everything. The model can ignore channels; it
cannot invent them.

VIS and SWIR are day-only, which is a modelling problem (the product must
work at night) but not a caching one -- `solar_elevation` is cached beside
them so the loader can gate on illumination rather than guessing from the
timestamp.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


# All six imager channels. See the module docstring: transfer is already
# paid, so the only question is storage, and storage is cheap.
ALL_CHANNELS = ("VIS", "SWIR", "MIR", "TIR1", "TIR2", "WV")

# Native resolutions, km. Recorded because they are NOT the same across
# satellites -- 3D/3DR put WV at 8 km and 3DS at 4 km, which is load-bearing
# and has already caused one wrong assumption (LIMITATIONS 4b).
NATIVE_KM = {"VIS": 1.0, "SWIR": 1.0, "MIR": 4.0,
             "TIR1": 4.0, "TIR2": 4.0, "WV": 8.0}


@dataclass(frozen=True)
class InsatCacheConfig:
    """Everything that changes the cached bytes. All of it is hashed."""

    # --- grid ---------------------------------------------------------------
    grid_shape: tuple = (912, 864)
    target_km: float = 4.0
    projection: str = "laea"
    proj_lat_0: float = 23.0
    proj_lon_0: float = 82.0
    resample: str = "nearest"
    radius_of_influence_m: float = 12000.0

    # --- satellite inputs ---------------------------------------------------
    channels: tuple = ALL_CHANNELS
    store_dtype: str = "uint16"        # scaled ints; float16 loses BT precision
    cadence_min: float = 30.0

    # --- per-scan metadata cached alongside the arrays ----------------------
    # Each of these has already caused, or would cause, a wrong number if
    # recovered later by assumption instead of stored now.
    per_scan_fields: tuple = (
        "scan_time_utc",            # provenance
        "satellite",                # 3D / 3DR / 3DS behave differently
        "sub_satellite_longitude",  # 74E vs 82E; parallax depends on it
        "solar_elevation",          # gates VIS/SWIR; the 3DS attribute is
                                    # unusable (denormal), so the DATASET value
        "reader",                   # satpy vs native LUT fallback
        "physics_check_passed",
    )

    # --- static channels (constant in time, versioned because they change) --
    static_channels: tuple = (
        "hand_m", "upa_km2", "flow_direction", "elevation_m",
        "curve_number", "hydrologic_soil_group", "landcover",
        "tan_zenith", "zenith_deg", "parallax_dy", "parallax_dx",
        "basin_id", "basin_area_km2",
    )
    static_source_versions: tuple = (
        "MERIT_Hydro_v1.0.1", "ESA_WorldCover_v200_2021",
        "HYSOGs250m_v1", "GHS_POP_R2023A",
    )

    # --- antecedent moisture: an INPUT, not an assumption --------------------
    # 15.3x between dry and wet on identical rain (80 mm on CN 70: 2.8 mm vs
    # 42.2 mm). Larger than the spread the rainfall model will produce, so a
    # fixed value is wrong in the under-forecasting direction on exactly the
    # saturated-catchment days floods happen. Derived from IMERG, so it needs
    # 5 days of history BEFORE the first cached scan of every event -- which
    # is an ingest-window decision, not just a channel decision.
    antecedent_channel: bool = True
    antecedent_days: int = 5
    antecedent_excludes_current_day: bool = True
    antecedent_source: str = "IMERG-Final-V07"
    antecedent_lead_in_days: int = 5   # extra days fetched before each event

    # --- labels -------------------------------------------------------------
    label_source: str = "IMERG-Final-V07"
    label_km: float = 12.0
    extreme_percentile: float = 99.9
    label_climatology: str = "per-cell-per-month"

    # --- policy -------------------------------------------------------------
    nan_policy: str = "MASK"
    store_finite_mask: bool = True

    # --- provenance ---------------------------------------------------------
    version: int = 1                   # bump to invalidate every cache
    notes: str = ""

    def __post_init__(self):
        unknown = set(self.channels) - set(NATIVE_KM)
        if unknown:
            raise ValueError(f"unknown channels: {sorted(unknown)}")
        if self.antecedent_channel and self.antecedent_lead_in_days < self.antecedent_days:
            raise ValueError(
                f"antecedent_lead_in_days ({self.antecedent_lead_in_days}) must "
                f"be >= antecedent_days ({self.antecedent_days}): the 5-day "
                f"window excludes the current day, so the first scan of an "
                f"event needs {self.antecedent_days} full days of IMERG before "
                f"it. Ingesting without that lead-in caches nan antecedent for "
                f"the start of every event -- which is the start of the storm.")

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self) | {"fingerprint": self.fingerprint()}

    def bytes_per_scan(self) -> int:
        cells = self.grid_shape[0] * self.grid_shape[1]
        width = 2 if self.store_dtype in ("uint16", "int16", "float16") else 4
        n = len(self.channels) + (1 if self.antecedent_channel else 0)
        mask = len(self.channels) if self.store_finite_mask else 0
        return cells * (n * width + mask)     # mask is 1 byte per channel

    def archive_gb(self, n_scans: int) -> float:
        return n_scans * self.bytes_per_scan() / 1024 ** 3

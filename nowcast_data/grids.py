"""The common analysis grid. Everything is resampled onto this.

4 km, because that is INSAT TIR-1's native resolution -- the finest thing
the operational feed actually resolves. Water vapour (8 km) and the IMERG
labels (~11 km) are coarser and sit on this grid carrying their own real
information content, exactly as SEVIR's channels were treated.

Projection: Lambert Azimuthal Equal Area centred on India. Equal-area
matters because the heads are scored with area-based metrics -- CSI, FSS
neighbourhoods in kilometres -- and a plate carree grid would make a
"4 km" cell mean something different in Kashmir than in Kerala, quietly
biasing every spatial score by latitude.

Dimensions are chosen so the arithmetic stays integral:
  864 / 4 = 216 and 912 / 4 = 228   -> model patch size 4
  864 / 12 = 72 and 912 / 12 = 76   -> 12 km IMERG-matched labels
"""
from __future__ import annotations

from dataclasses import dataclass

RESOLUTION_M = 4000.0
SHAPE = (912, 864)          # (rows, cols) = (y, x)
CENTRE_LAT, CENTRE_LON = 23.0, 82.0

_HALF_X = SHAPE[1] * RESOLUTION_M / 2
_HALF_Y = SHAPE[0] * RESOLUTION_M / 2
EXTENT = (-_HALF_X, -_HALF_Y, _HALF_X, _HALF_Y)   # metres, (xmin,ymin,xmax,ymax)

PROJ = {"proj": "laea", "lat_0": CENTRE_LAT, "lon_0": CENTRE_LON,
        "datum": "WGS84", "units": "m"}


def india_area():
    """pyresample AreaDefinition for the common grid."""
    from pyresample.geometry import AreaDefinition
    return AreaDefinition(
        area_id="india_4km",
        description="India 4 km LAEA -- common analysis grid",
        proj_id="india_laea",
        projection=PROJ,
        width=SHAPE[1], height=SHAPE[0],
        area_extent=EXTENT,
    )


@dataclass(frozen=True)
class GridSpec:
    resolution_m: float = RESOLUTION_M
    shape: tuple[int, int] = SHAPE

    @property
    def resolution_km(self) -> float:
        return self.resolution_m / 1000.0

    def divisible_by(self, km: float) -> bool:
        """Can this grid carry labels at `km` without fractional blocks?"""
        f = km / self.resolution_km
        if abs(f - round(f)) > 1e-9:
            return False
        n = int(round(f))
        return self.shape[0] % n == 0 and self.shape[1] % n == 0


INDIA_4KM = GridSpec()


def bounds_lonlat():
    """(lon_min, lat_min, lon_max, lat_max) of the grid corners."""
    area = india_area()
    lons, lats = area.get_lonlats()
    return (float(lons.min()), float(lats.min()),
            float(lons.max()), float(lats.max()))

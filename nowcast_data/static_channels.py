"""Static per-cell channels: viewing geometry, and later terrain.

Why satellite zenith angle is an input channel
----------------------------------------------
A cloud top is not above the rain it produces. Parallax displaces it away
from the sub-satellite point by h*tan(zenith) -- measured for INSAT-3DR at
74E with a 12 km top: 0.5 px at Kanyakumari, 2.0 at Delhi, 2.5 at Kashmir.
So the model must learn a LATITUDE-DEPENDENT displacement between INSAT
features and IMERG labels, and it is worst over exactly the cloudburst
region.

Two ways to handle that. We take the second.

**Deprojecting cloud tops to the surface in preprocessing** -- rejected, on
one structural objection and three practical ones:

  * Different clouds in the SAME scene sit at different heights, so they
    shift by different amounts. There is no single warp to apply: you need a
    per-pixel displacement field, which tears holes where nothing maps and
    collides where two pixels map to one. That is not a tuning problem, it
    is the wrong shape of operation.
  * It needs a cloud height we do not have. CTT -> height requires a
    temperature profile (ERA5 could supply one) and is ambiguous for thin
    cirrus, multi-layer cloud and inversions.
  * It is irreversible: bake in a wrong height and the original is gone.
  * It must be repeated at inference, adding latency to the claim that
    latency is our advantage.

**Satellite zenith as a static input channel** -- taken:

  * The model ALREADY has everything the correction needs. Displacement is
    h*tan(z): it reads tan(z) from this channel, and h is implicit in cloud
    top temperature, which is already an input. Deprojection would perform
    that computation FOR the model using a worse height estimate than the
    model can infer from the full radiance context.
  * Static: computed once per grid, zero inference cost.
  * Reversible -- it is only an input.
  * Composes with the DEM channel, which is static for the same reason.

We supply **tan(zenith)**, not the angle. The physics is linear in tan(z),
so the network can express the correction as a linear function of the
channel instead of having to learn a tangent.

Cost, stated: this spends a little model capacity on a correction we could
have applied ourselves. That is the right trade against a preprocessing step
that tears the image and assumes a height.

Receptive field: the displacement is at most ~2.5 cells, well inside a
patch-4 backbone with attention, so the model can see both the cloud top and
the rain beneath it.

PROVENANCE: PURE GEOMETRY, NOTHING FITTED
-----------------------------------------
Every input to `zenith_channels` is the grid's own lat/lon, a published
sub-satellite longitude, and spherical trigonometry. No coefficient here was
tuned against any measurement, and in particular none against the
cloud-to-rain regression that LIMITATIONS 15 invalidated. The only free
number in this module is `cloud_top_km=12.0`, and it lives in
`expected_shift_px`, a reporting helper the cache never calls.

WHAT THE CHANNEL CANNOT DO, AND WHY IT STILL GOES IN
-----------------------------------------------------
tan(zenith) is 95% collinear with LATITUDE over India from 74E
(corr = +0.948, LIMITATIONS 15). Two consequences, both worth stating before
this is baked into a 197 GB cache:

  * The channel gives the model little that a position encoding does not
    already carry on 3DR-only data. It is still correct, still static, still
    free at inference -- but it is not the independent information the
    docstring above implies when only one satellite is in the training set.
  * A post-hoc registration check can measure THAT predictions are displaced.
    It cannot attribute the displacement to parallax rather than to anything
    else varying with latitude, for exactly the reason the gate could not.
    Attribution needs 3DS (82E) and 3DR (74E) over the same ground.

Neither is a reason to drop the channel: it costs nothing and it is right.
Both are reasons not to claim the model "learned parallax" from 3DR alone.
"""
from __future__ import annotations

import numpy as np

from .alignment_gate import satellite_zenith
from .grids import india_area

# INSAT-3DR. 3D was 82E and 3DS is 82E -- the sub-longitude differs per
# satellite, so this channel is satellite-specific and must be rebuilt if the
# training satellite changes.
INSAT_3DR_SUB_LON = 74.0
INSAT_3D_SUB_LON = 82.0
INSAT_3DS_SUB_LON = 82.0


def zenith_channels(sub_lon: float = INSAT_3DR_SUB_LON, area=None) -> dict:
    """tan(zenith) and its two components, on the common India grid.

    Returns:
      tan_zenith      the parallax scale factor -- displacement = h * this
      zenith_deg      the raw angle, for diagnostics and reporting
      parallax_dy     unit northward displacement per km of cloud height
      parallax_dx     unit eastward  displacement per km of cloud height

    The two components matter because parallax has a DIRECTION: away from the
    sub-satellite point. Supplying only the magnitude would leave the model to
    infer the bearing from position, which is the inference we are trying to
    save it.
    """
    area = area or india_area()
    lons, lats = area.get_lonlats()
    z = satellite_zenith(lats, lons, sub_lon=sub_lon)
    tz = np.tan(np.deg2rad(z))

    # Bearing from the sub-satellite point toward each cell, on the sphere.
    la = np.deg2rad(lats)
    dlo = np.deg2rad(lons - sub_lon)
    y = np.sin(dlo) * np.cos(la)                       # eastward component
    x = np.cos(0.0) * np.sin(la) - np.sin(0.0) * np.cos(la) * np.cos(dlo)
    norm = np.hypot(x, y)
    norm = np.where(norm > 1e-9, norm, 1.0)

    return {
        "tan_zenith": tz.astype(np.float32),
        "zenith_deg": z.astype(np.float32),
        "parallax_dy": (tz * x / norm).astype(np.float32),
        "parallax_dx": (tz * y / norm).astype(np.float32),
    }


def expected_shift_px(cloud_top_km: float = 12.0, grid_km: float = 4.0,
                      sub_lon: float = INSAT_3DR_SUB_LON) -> np.ndarray:
    """Parallax shift in grid cells, for reporting and for the gate."""
    return zenith_channels(sub_lon)["tan_zenith"] * cloud_top_km / grid_km

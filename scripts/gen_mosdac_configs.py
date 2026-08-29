"""Generate the MOSDAC download configs. Re-run to regenerate.

Shape follows the user's VERIFIED-WORKING 2023 config exactly, which the API
manual also confirms: user_credentials / search_parameters / download_settings,
startTime+endTime as YYYY-MM-DD, boundingBox as minLon,minLat,maxLon,maxLat.
"""
import json
from pathlib import Path

OUT = Path("configs/mosdac")
OUT.mkdir(parents=True, exist_ok=True)

# India 4 km analysis grid corner bounds, LONGITUDE FIRST.
INDIA_BBOX = "62.3,5.8,101.7,39.5"

# Catalog-verified, read from mosdac.gov.in/catalog/satellite.php on 2026-08-29.
#   3DIMG_L1B_STD  2013-10-01 .. 2024-06-18  IN-ACTIVE
#   3RIMG_L1B_STD  2016-10-11 .. present     ACTIVE
#   3SIMG_L1B_STD  2024-05-17 .. present     ACTIVE
EVENTS = [
    ("chennai-floods",      "2015-12-01", "2015-12-02", "3DIMG_L1B_STD",
     "pre-3DR: INSAT-3D is the ONLY source this far back"),
    ("mumbai-deluge",       "2017-08-29", "2017-08-29", "3RIMG_L1B_STD", ""),
    ("kerala-floods",       "2018-08-15", "2018-08-16", "3RIMG_L1B_STD",
     "orographic -- the DEM case"),
    ("kerala-floods-b",     "2018-08-17", "2018-08-17", "3RIMG_L1B_STD", ""),
    ("hyderabad-urban",     "2020-10-13", "2020-10-14", "3RIMG_L1B_STD", ""),
    ("uttarakhand",         "2021-10-17", "2021-10-18", "3RIMG_L1B_STD", ""),
    ("uttarakhand-b",       "2021-10-19", "2021-10-19", "3RIMG_L1B_STD", ""),
    ("amarnath-cloudburst", "2022-07-08", "2022-07-08", "3RIMG_L1B_STD",
     "true cloudburst, station-scale"),
    ("bengaluru-urban",     "2022-09-05", "2022-09-05", "3RIMG_L1B_STD", ""),
    ("himachal",            "2023-07-08", "2023-07-09", "3RIMG_L1B_STD", ""),
    ("himachal-b",          "2023-07-10", "2023-07-11", "3RIMG_L1B_STD", ""),
    ("wayanad-landslide",   "2024-07-30", "2024-07-30", "3RIMG_L1B_STD",
     "3DIMG is In-Active after 2024-06-18 -- must use 3RIMG or 3SIMG"),
]


def cfg(dataset, start, end, bbox, count="", note="", unblocks=""):
    return {
        "_note": note,
        "_unblocks": unblocks,
        "user_credentials": {"username": "", "password": ""},
        "search_parameters": {
            "datasetId": dataset,
            "startTime": start,
            "endTime": end,
            "count": count,
            "boundingBox": bbox,
            "gId": "",
        },
        "download_settings": {
            "download_path": "data/insat",
            "organize_by_date": True,
            "skip_user_prompt": False,
            "generate_error_log": True,
            "error_log_path": "data/insat/logs",
        },
    }


files = []

# --- 00/01: format verification, one scan each, FULL DISK ------------------
files.append(("00_verify_3DS_current.json", cfg(
    "3SIMG_L1B_STD", "2026-08-01", "2026-08-01", "", count=1,
    note="Format check on the CURRENT operational satellite (INSAT-3DS). "
         "Full disk deliberately: a spatial subset could mask a geometry change.",
    unblocks="Everything. Do not bulk-download before this passes ingest_scan.")))

files.append(("01_verify_3DR_current.json", cfg(
    "3RIMG_L1B_STD", "2026-08-01", "2026-08-01", "", count=1,
    note="Format check on INSAT-3DR separately. 3DR is a DIFFERENT satellite "
         "from 3DS -- a reader that handles one is not proven on the other, and "
         "3DR is the workhorse for every hindcast event from 2016 onward.",
    unblocks="All 3RIMG event windows (02-11).")))

files.append(("02_verify_3D_archive.json", cfg(
    "3DIMG_L1B_STD", "2015-12-01", "2015-12-01", "", count=1,
    note="Format check on the INSAT-3D ARCHIVE. In-Active since 2024-06-18 but "
         "still the only source before 2016-10-11. We hold a 2019 3DIMG file "
         "that already passes, so this mainly confirms archive retrieval works.",
    unblocks="The Chennai 2015 hindcast case.")))

# --- 03+: hindcast events --------------------------------------------------
for i, (name, s, e, ds, note) in enumerate(EVENTS, start=3):
    files.append((f"{i:02d}_event_{name}.json", cfg(
        ds, s, e, INDIA_BBOX,
        note=note or f"Hindcast event window: {name}.",
        unblocks="Indian hindcast validation -- the cases judges ask about by name.")))

# --- training archive: one config per half-month of monsoon ---------------
n = len(files) + 1
for year in (2023, 2024, 2025):
    for (ms, me) in [("06-01", "06-15"), ("06-16", "06-30"), ("07-01", "07-15"),
                     ("07-16", "07-31"), ("08-01", "08-15"), ("08-16", "08-31"),
                     ("09-01", "09-15"), ("09-16", "09-30")]:
        ds = "3RIMG_L1B_STD"
        files.append((f"{n:02d}_archive_{year}_{ms}_{me}.json", cfg(
            ds, f"{year}-{ms}", f"{year}-{me}", INDIA_BBOX,
            note=f"Monsoon training archive, {year} {ms}..{me}. 3RIMG spans "
                 f"2016-10-11 to present, so it covers all three seasons with "
                 f"ONE satellite -- avoiding a sensor discontinuity mid-training.",
            unblocks="Indian fine-tuning set. Only after the hindcast path works.")))
        n += 1

for fn, c in files:
    (OUT / fn).write_text(json.dumps(c, indent=2) + "\n")
print(f"wrote {len(files)} configs to {OUT}/")
for fn, _ in files[:6]:
    print("   ", fn)
print(f"    ... and {len(files)-6} more")

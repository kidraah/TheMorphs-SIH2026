"""RUN THIS FIRST when MOSDAC comes back, before pulling anything else.

It answers the one question left about the 3.4 TB archive. The HDF5-layout
half is already answered on a real delivered scan, served locally:

    482.6 MB file, 8 datasets fetched, 18.9 MB transferred, 111 reads
    25.5x saving, byte-identical to the local file

so if MOSDAC honours Range the archive is 141 GB instead of 3,596 GB, and
5.3 hours instead of 135.8 at the measured 7.9 MB/s.

Three steps, in order, and step 2 is the one that matters:

  1. HEAD -- does it advertise Accept-Ranges? (a claim, not evidence)
  2. a real ranged GET -- 206, correct Content-Range, exact byte count
  3. fetch only the kept datasets and diff them against a full download of
     the SAME scan, byte for byte

Step 3 is not optional. A wrong chunk index returns the right shape and
plausible values, and that is the failure this project keeps finding.

    python scripts/probe_mosdac_range.py --url "<download url>" \
        --local /path/to/the/same/scan.h5

Credentials come from the gitignored .env via nowcast_data.credentials --
none are read from or written to any tracked file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Kept datasets: the six imager channels plus the geolocation and LUTs each
# needs. Matches InsatCacheConfig.channels -- if that changes, this changes.
KEEP = [
    "IMG_VIS", "IMG_SWIR", "IMG_MIR", "IMG_TIR1", "IMG_TIR2", "IMG_WV",
    "IMG_TIR1_TEMP", "IMG_TIR2_TEMP", "IMG_MIR_TEMP", "IMG_WV_TEMP",
    "IMG_VIS_ALBEDO", "IMG_SWIR_RADIANCE",
    "Latitude", "Longitude", "Latitude_WV", "Longitude_WV",
    "Latitude_VIS", "Longitude_VIS",
    "Sun_Elevation", "Sat_Elevation", "GreyCount", "SCAN_LINE_TIME",
]
# Without VIS/SWIR and their int32 geolocation -- what the model reads today.
KEEP_MINIMAL = [
    "IMG_MIR", "IMG_TIR1", "IMG_TIR2", "IMG_WV",
    "IMG_TIR1_TEMP", "IMG_TIR2_TEMP", "IMG_MIR_TEMP", "IMG_WV_TEMP",
    "Latitude", "Longitude", "Latitude_WV", "Longitude_WV",
    "Sun_Elevation", "GreyCount",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="a MOSDAC download URL")
    ap.add_argument("--local", default="",
                    help="the SAME scan already on disk, for the byte diff")
    ap.add_argument("--minimal", action="store_true",
                    help="fetch only the non-VIS datasets")
    ap.add_argument("--scans", type=int, default=8000)
    ap.add_argument("--rate", type=float, default=7.9,
                    help="MB/s; 7.9 is measured from the event pull")
    ap.add_argument("--out", default="runs/mosdac_range_probe.json")
    a = ap.parse_args()

    import requests

    from nowcast_data.ranged import (probe_range_support, read_datasets,
                                     verify_against_local)

    # Auth: a bearer token from the gitignored .env. The working downloader
    # has a refresh-token flow (visible in its logs); that flow is NOT
    # reimplemented here, because guessing at someone else's auth handshake
    # is how you end up probing an error page and calling it a 206. Paste a
    # current token, or point --url at an already-authorised URL.
    from nowcast_data.credentials import get, load_env
    load_env()
    session = requests.Session()
    token = get("MOSDAC_TOKEN", required=False)
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
        print("  using MOSDAC_TOKEN from .env")
    else:
        print("  no MOSDAC_TOKEN in .env -- probing unauthenticated. If the "
              "server answers with a login page this probe is meaningless; "
              "check the byte count, not just the status code.")

    names = KEEP_MINIMAL if a.minimal else KEEP

    print("STEP 1-2: does the server really honour Range?")
    sup = probe_range_support(a.url, session)
    print(sup.report())
    result = {"url_host": a.url.split("/")[2] if "//" in a.url else "",
              "range": sup.__dict__}

    if not sup.conclusive:
        print("\n  INCONCLUSIVE -- the probe never reached a file body, so it "
              "says nothing about Range.")
        print("  Do NOT size storage on this. Put MOSDAC_TOKEN (or a working "
              "session) in the gitignored .env and re-run.")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2, default=str))
        return 3

    if not sup.supported:
        full = a.scans * 448.0 / 1024
        print(f"\n  Range NOT usable. The archive stays at {full:,.0f} GB "
              f"({full*1024/a.rate/3600:.0f} h at {a.rate} MB/s).")
        print("  Streaming decode-and-discard is then the plan: raw must "
              "never accumulate, only the 117 GB decoded cache.")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2, default=str))
        return 1

    print(f"\nSTEP 3: fetch {len(names)} datasets and diff against the local copy")
    if a.local:
        res = verify_against_local(a.url, a.local, names, session)
        print(res["report"])
        print(f"  byte-identical: {res['passed']}  "
              f"({res['datasets_checked']} datasets, {res['n_requests']} reads)")
        if res["mismatches"]:
            for m in res["mismatches"][:10]:
                print(f"    !! {m}")
        result["verify"] = {k: v for k, v in res.items() if k != "mismatches"}
        result["mismatches"] = res["mismatches"]
        per_scan = res["bytes_transferred"]
        full_scan = res["file_size"]
        ok = res["passed"]
    else:
        r = read_datasets(a.url, names, session)
        print(r.report())
        print("  !! no --local given, so NOTHING was verified byte-for-byte. "
              "A wrong chunk index returns the right shape and plausible "
              "values. Re-run with --local before trusting this.")
        per_scan, full_scan, ok = r.bytes_transferred, None, False
        result["read"] = {"bytes": r.bytes_transferred, "n": r.n_requests}

    gb = a.scans * per_scan / 1024 ** 3
    print(f"\n  archive projection, {a.scans:,} scans:")
    if full_scan:
        fg = a.scans * full_scan / 1024 ** 3
        print(f"    without Range : {fg:>8,.0f} GB   "
              f"{fg*1024/a.rate/3600:>6.1f} h at {a.rate} MB/s")
    print(f"    with Range    : {gb:>8,.0f} GB   "
          f"{gb*1024/a.rate/3600:>6.1f} h at {a.rate} MB/s")
    result["projection_gb"] = gb

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2, default=str))
    print(f"\nwrote {a.out}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

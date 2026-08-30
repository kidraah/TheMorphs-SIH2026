"""Size the event-sampled INSAT archive.

Continuous monsoon coverage is ~15 TB and overwhelmingly non-convective: at a
3.8e-4 base rate most of it is the model watching nothing happen. SEVIR --
which works -- is ~12,000 EVENTS drawn from 526 storm days, not years of
continuous data.

So: rank days by IMERG convective activity, take the top N, and add a
DELIBERATE sample of null days so the base rate stays honest and the null
test set from the verification protocol survives.
"""
import argparse

# Measured on 702 real event files (292 GB) delivered by MOSDAC with a
# boundingBox in the request:
#
#     3RIMG  611 files  mean 448.0 MB  p50 447.4  min 57.7  max 500.9
#     3DIMG   91 files  mean 434.4 MB  p50 428.9
#
# THE BOUNDINGBOX IS A SEARCH FILTER, NOT A SERVER-SIDE SUBSET. Confirmed
# from the data rather than inferred from the size: a delivered file's
# Latitude spans -81.04 to +81.04 and Longitude -7.15 to +155.15, which is
# the full Earth disk from 82E, not an India crop (6-39N, 62-102E).
#
# So BBOX_AREA_FRACTION does not apply to what is transferred. It is kept
# below only to show what the plan assumed.
FULL_DISK_MB = 430.0
MEASURED_DELIVERED_MB = 448.0

# What is actually inside those 448 MB, by stored bytes in one real file:
#     Longitude_VIS   139.1 MB  30.8%   <- not used
#     Latitude_VIS     87.9 MB  19.5%   <- not used
#     IMG_SWIR         86.4 MB  19.2%   <- not used
#     IMG_VIS          80.5 MB  17.8%   <- not used
#     everything else  57.3 MB  12.7%
# The channels the model reads are 19.1 MB, 4.2% of the file. 95.8% of every
# byte transferred is 1 km VIS/SWIR and their int32 geolocation.
USEFUL_MB = 19.1

# India grid, uint16 scaled, per channel per scan.
GRID_CELLS = 864 * 912
# India box is ~10% of the visible disk by area. The old plan ASSUMED the
# bbox subset scaled with that. RESOLVED, and wrong: 702 delivered files are
# full-disk. Kept only to show what the estimate was and by how much it
# missed -- the dominant uncertainty in the plan is now a measured fact.
BBOX_AREA_FRACTION = 0.10

ap = argparse.ArgumentParser()
ap.add_argument("--active-days", type=int, default=300)
ap.add_argument("--null-days", type=int, default=100)
# KEEP vs TRANSFER. These two are how many scans we WANT per day. They are
# NOT how many are downloaded: MOSDAC's search selects by date range and has
# no time-of-day filter, so the smallest downloadable unit is a whole day --
# all 48 scans. Sizing the pull from these numbers is a category error, and
# it is the same one made by `boundingBox` (a keep-shaped filter assumed to
# subset the transfer, measured 10.4x low) and by the channel list (2 of 6
# channels assumed to bound bytes fetched, measured 23.5x low).
#
# One category error, three terms, compounding to 25x:
#     planned:  8,000 scans x  43 MB =   336 GB
#     actual : 19,200 scans x 448 MB = 8,400 GB
SCANS_PER_DAY_DOWNLOADED = 48

ap.add_argument("--scans-per-active-day", type=int, default=24,
                help="30-min cadence over a 12h convective window; 3DR scans :15 and :45 so 48/day is available if wanted")
ap.add_argument("--scans-per-null-day", type=int, default=8)
a = ap.parse_args()

per_scan_gb = MEASURED_DELIVERED_MB / 1024
estimated_gb = FULL_DISK_MB * BBOX_AREA_FRACTION / 1024

print("EVENT-SAMPLED ARCHIVE\n")
print(f"  per-scan transferred (MEASURED):  {per_scan_gb*1024:.0f} MB")
print(f"  per-scan assumed by the old plan:  {estimated_gb*1024:.0f} MB "
      f"({FULL_DISK_MB:.0f} MB x {BBOX_AREA_FRACTION:.0%} area)")
print(f"  the estimate was low by {per_scan_gb/estimated_gb:.1f}x -- "
      f"MOSDAC does not subset")
print(f"    RESOLVED on 702 real files: MOSDAC does not subset server-side\n")

rows = [("active", a.active_days, a.scans_per_active_day),
        ("null",   a.null_days,   a.scans_per_null_day)]
tot_scans = tot_gb = 0
print(f"  {'class':>8} {'days':>6} {'scans/day':>10} {'scans':>7} {'GB':>8}")
for name, days, spd in rows:
    n = days * spd
    gb = n * per_scan_gb
    tot_scans += n; tot_gb += gb
    print(f"  {name:>8} {days:>6} {spd:>10} {n:>7,} {gb:>8.0f}")
print(f"  {'TOTAL':>8} {a.active_days+a.null_days:>6} {'':>10} {tot_scans:>7,} {tot_gb:>8.0f}")

print(f"\n  vs continuous 3 seasons: ~15,000 GB  ->  {15000/max(tot_gb,1):.1f}x smaller")
print(f"  null fraction: {a.null_days/(a.active_days+a.null_days):.0%} of days, "
      f"{a.null_days*a.scans_per_null_day/tot_scans:.0%} of scans")
print("\n  at 20 MB/s sustained: "
      f"{tot_gb*1024/20/3600:.1f} hours  ({tot_gb*1024/20/86400:.1f} days)")

print("\nWHY BREADTH, NOT CONTINUITY")
print("  The binding constraint is INDEPENDENT EPISODES, not volume. SEVIR's")
print("  test split had 9. Spreading 300 active days across 2017-2025 buys")
print("  many more independent synoptic episodes than the same 300 days taken")
print("  from one or two seasons, where consecutive days share a setup.")

print("\nWHY NULL DAYS ARE SAMPLED, NOT DROPPED")
print("  Training only on active days inflates the base rate the model sees,")
print("  so it over-forecasts in operation -- and the harness's null test set")
print("  (docs: ~200 null days) has nothing to score against. The null sample")
print("  is deliberate and documented so the base rate stays honest.")


# ---------------------------------------------------------------------------
# Transfer is not storage. Decode-and-discard separates them.
# ---------------------------------------------------------------------------
print("\n" + "=" * 66)
print("KEEP vs TRANSFER -- the day granularity")
print("=" * 66)
dl_scans = (a.active_days + a.null_days) * SCANS_PER_DAY_DOWNLOADED
print(f"  scans we want to KEEP:      {tot_scans:>8,}")
print(f"  scans we must DOWNLOAD:     {dl_scans:>8,}  "
      f"({a.active_days + a.null_days} days x {SCANS_PER_DAY_DOWNLOADED}/day)")
print(f"  MOSDAC has no time-of-day filter, so a day is the smallest unit.")
print(f"  Sizing the pull from the keep figure understates it "
      f"{dl_scans/max(tot_scans,1):.1f}x.")
tot_scans = dl_scans

print("\n" + "=" * 66)
print("TRANSFER vs STORAGE")
print("=" * 66)
transfer_gb = tot_scans * MEASURED_DELIVERED_MB / 1024
print(f"  raw transferred, unavoidable:      {transfer_gb:>8,.0f} GB "
      f"({transfer_gb/1024:.1f} TB)")
print(f"  raw STORED if kept:                {transfer_gb:>8,.0f} GB")
for nch, label in ((2, "TIR1+WV only"), (6, "all 6 IR/VIS channels")):
    gb = tot_scans * GRID_CELLS * nch * 2 / 1024**3
    print(f"  decoded to the India grid, {nch} ch:  {gb:>8,.0f} GB   ({label})")
print(f"\n  Every byte of VIS/SWIR is transferred whether or not it is kept,")
print(f"  so extracting ALL channels costs ~{tot_scans*GRID_CELLS*4*2/1024**3:.0f} GB more storage and")
print(f"  saves a {transfer_gb/1024:.1f} TB re-download if a channel is wanted later.")
print(f"  Decode-and-discard is what makes the plan viable: raw never")
print(f"  accumulates, only the decoded cache does.")

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

# Measured: a full-disk 3RIMG L1B scan is ~430 MB (three real files).
FULL_DISK_MB = 430.0
# India box is ~10% of the visible disk by area. The bbox subset is assumed to
# scale with area -- UNVERIFIED, because MOSDAC is down and we have never
# received a bbox'd file. This is the dominant uncertainty in the estimate.
BBOX_AREA_FRACTION = 0.10

ap = argparse.ArgumentParser()
ap.add_argument("--active-days", type=int, default=300)
ap.add_argument("--null-days", type=int, default=100)
ap.add_argument("--scans-per-active-day", type=int, default=24,
                help="30-min cadence over a 12h convective window; 3DR scans :15 and :45 so 48/day is available if wanted")
ap.add_argument("--scans-per-null-day", type=int, default=8)
a = ap.parse_args()

per_scan_gb = FULL_DISK_MB * BBOX_AREA_FRACTION / 1024

print("EVENT-SAMPLED ARCHIVE\n")
print(f"  per-scan size (bbox subset, ESTIMATED): {per_scan_gb*1024:.0f} MB")
print(f"    from {FULL_DISK_MB:.0f} MB full disk x {BBOX_AREA_FRACTION:.0%} area")
print(f"    UNVERIFIED -- confirm on the first bbox'd file; the whole estimate scales with it\n")

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

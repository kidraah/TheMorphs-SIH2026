#!/usr/bin/env bash
# Fetch the SEVIR channels this project uses. ~228 GiB.
#
# ir069 (water vapour)   45.6 GiB  -> stands in for INSAT WV, degraded to 8 km
# ir107 (thermal IR)     45.6 GiB  -> stands in for INSAT TIR, 4 km
# vil   (NEXRAD VIL)    137.2 GiB  -> pretraining label
#
# vis and lght are deliberately skipped: vis is daytime-only (a night-capable
# model cannot rely on it) and lght has a different, non-gridded storage layout.
#
# `aws s3 sync` is resumable -- rerun after an interruption and it skips what
# is already present.
set -euo pipefail
DEST="${1:-/Users/evad/nowcast/data/sevir}"
mkdir -p "$DEST/data"

echo "[$(date +%H:%M:%S)] catalog"
aws s3 cp --no-sign-request s3://sevir/CATALOG.csv "$DEST/CATALOG.csv"

for t in ir069 ir107 vil; do
  echo "[$(date +%H:%M:%S)] $t"
  aws s3 sync --no-sign-request --only-show-errors \
      "s3://sevir/data/$t/" "$DEST/data/$t/"
  echo "[$(date +%H:%M:%S)] $t done: $(du -sh "$DEST/data/$t" | cut -f1)"
done
echo "[$(date +%H:%M:%S)] ALL DONE: $(du -sh "$DEST" | cut -f1)"

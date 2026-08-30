"""
scripts/ — Pre-training data preparation utilities.

Run these ONCE before training whenever the source labels change:
    python scripts/regenerate_cb_labels.py   # CB from HEM (1km resolution)
    python scripts/regenerate_ff_labels.py   # FF from CB + DEM slope

After training:
    python scripts/upload_to_hf.py           # Upload checkpoint to HuggingFace
"""

"""Credential handling: gitignored .env, read through environment variables.

Nothing here ever prints or logs a secret value, and .env is gitignored.
Credentials are read at call time rather than captured at import, so a
populated .env takes effect without restarting a long-running process.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

REQUIRED = {
    "MOSDAC_USERNAME": "MOSDAC account (mosdac.gov.in) -- INSAT-3D/3DR",
    "MOSDAC_PASSWORD": "MOSDAC password",
    "EARTHDATA_USERNAME": "NASA Earthdata Login -- GPM IMERG via GES DISC",
    "EARTHDATA_PASSWORD": "NASA Earthdata password",
}
OPTIONAL = {
    "NCMRWF_USERNAME": "NCMRWF RDS -- IMDAA reanalysis (deferred; ERA5 covers "
                       "thermodynamics for now)",
    "NCMRWF_PASSWORD": "NCMRWF password",
    "CDSAPI_KEY": "Copernicus CDS -- ERA5",
}


def load_env(path: Path | None = None, override: bool = False) -> int:
    """Read KEY=VALUE lines from .env into os.environ. Returns count loaded.

    Deliberately minimal: no interpolation, no shell expansion, no export
    semantics. Existing environment variables win unless `override`, so a CI
    secret store is not silently shadowed by a stale local file.
    """
    p = Path(path or ENV_FILE)   # coerce the fallback too, not just the argument
    if not p.exists():
        return 0
    n = 0
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = val
            n += 1
    return n


def get(name: str, required: bool = True) -> str | None:
    """Fetch one credential. Raises with guidance rather than a KeyError."""
    load_env()
    val = os.environ.get(name)
    if val:
        return val
    if not required:
        return None
    desc = REQUIRED.get(name) or OPTIONAL.get(name) or name
    raise RuntimeError(
        f"{name} is not set ({desc}).\n"
        f"Add it to {Path(ENV_FILE)} (gitignored) as `{name}=...` or export it.\n"
        f"See .env.example.")


def status() -> dict[str, bool]:
    """Which credentials are present. Never returns the values themselves."""
    load_env()
    return {k: bool(os.environ.get(k)) for k in (*REQUIRED, *OPTIONAL)}


def report() -> str:
    st = status()
    lines = ["credential status (values never shown)", ""]
    for k, desc in REQUIRED.items():
        lines.append(f"  [{'x' if st[k] else ' '}] {k:22s} {desc}")
    lines.append("")
    for k, desc in OPTIONAL.items():
        lines.append(f"  [{'x' if st[k] else ' '}] {k:22s} (optional) {desc}")
    missing = [k for k in REQUIRED if not st[k]]
    if missing:
        lines += ["", f"missing {len(missing)} required: {', '.join(missing)}",
                  f"populate {ENV_FILE}"]
    return "\n".join(lines)

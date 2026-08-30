"""MOSDAC auth, lifted from the user's working mdapi.py rather than guessed.

Guessing at someone else's handshake is how you probe an error page and call
it a 206 -- the unauthenticated probe already returned
{"code":"NO_ACCESS_TOKEN"} and an earlier version of the checker read that as
"Range not supported", which would have justified provisioning 3.4 TB.

The flow, verbatim from mdapi.py:

    POST /download_api/gettoken     {"username","password"}
                                 -> {"access_token","refresh_token"}
    GET  /download_api/download?id=<record_id>
                                    Authorization: Bearer <access_token>
    401 + {"code":"INVALID_TOKEN"}
      -> POST /download_api/refresh-token {"refresh_token"}
                                 -> {"access_token","refresh_token"}
    POST /download_api/logout       {"username"}

Search needs no auth:

    GET https://mosdac.gov.in/apios/datasets.json
        ?datasetId&startTime&endTime&count&boundingBox&gId&startIndex
     -> {"totalResults", "totalSizeMB", "itemsPerPage", "entries":[...]}

`totalResults` and `totalSizeMB` are the AUTHORITATIVE size of a config,
returned before anything is downloaded. That is strictly better than the
date-arithmetic estimate in mosdac_config, and it is what produced the
"179,134 files, 69.70 TB" prompt.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

TOKEN_URL = "https://mosdac.gov.in/download_api/gettoken"
REFRESH_URL = "https://mosdac.gov.in/download_api/refresh-token"
LOGOUT_URL = "https://mosdac.gov.in/download_api/logout"
DOWNLOAD_URL = "https://mosdac.gov.in/download_api/download"
SEARCH_URL = "https://mosdac.gov.in/apios/datasets.json"

# From mdapi.py's 429 handling: the server enforces BOTH a per-minute and a
# per-DAY limit, and the daily one is fatal (it logs out and exits). A daily
# cap may bind the archive plan harder than bandwidth does -- 8.2 TB at
# 7.9 MB/s is 289 h, but at N files/day it could be far longer. The number is
# not in mdapi.py; it comes back in the 429 body.
RATE_LIMIT_TYPES = ("minute_limit", "daily_limit")


class AuthError(RuntimeError):
    pass


@dataclass
class Session:
    access_token: str = ""
    refresh_token: str = ""
    username: str = ""
    refreshes: int = field(default=0)

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.access_token}"}


def login(username: str | None = None, password: str | None = None,
          timeout: float = 60.0) -> Session:
    import requests

    u = username or os.environ.get("MOSDAC_USERNAME", "")
    p = password or os.environ.get("MOSDAC_PASSWORD", "")
    if not u or not p:
        raise AuthError("MOSDAC_USERNAME / MOSDAC_PASSWORD not set (.env)")

    r = requests.post(TOKEN_URL, json={"username": u, "password": p},
                      timeout=timeout)
    if r.status_code in (400, 401):
        try:
            msg = r.json().get("error", r.text[:200])
        except Exception:
            msg = r.text[:200]
        raise AuthError(f"login rejected ({r.status_code}): {msg}")
    if r.status_code == 503:
        raise AuthError("MOSDAC reports service unavailable (503)")
    r.raise_for_status()
    j = r.json()
    return Session(j.get("access_token", ""), j.get("refresh_token", ""), u)


def refresh(session: Session, timeout: float = 60.0) -> Session:
    """Refresh in place. This is the handshake that killed the event pull."""
    import requests

    r = requests.post(REFRESH_URL, json={"refresh_token": session.refresh_token},
                      timeout=timeout)
    if r.status_code == 400:
        raise AuthError(f"refresh rejected: {r.json().get('error', r.text[:200])}")
    r.raise_for_status()
    j = r.json()
    session.access_token = j.get("access_token", "")
    session.refresh_token = j.get("refresh_token", "")
    session.refreshes += 1
    return session


def logout(session: Session, timeout: float = 30.0) -> bool:
    import requests

    try:
        r = requests.post(LOGOUT_URL, json={"username": session.username},
                          timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def search(params: dict, timeout: float = 120.0) -> dict:
    """Authoritative count and size for a config. No auth, no data transfer."""
    import requests

    data = {k: v for k, v in params.items() if v not in ("", None)}
    r = requests.get(SEARCH_URL, params=data, timeout=timeout)
    r.raise_for_status()
    return r.json()


def download_url(record_id) -> str:
    return f"{DOWNLOAD_URL}?id={record_id}"

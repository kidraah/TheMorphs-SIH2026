"""Fetch only the HDF5 datasets we keep, over HTTP Range requests.

The prize
---------
MOSDAC delivers full Earth disk: 448 MB per scan, of which the datasets we
cache are 19.1 MB -- 4.2%. HDF5 is a random-access format with a B-tree
index, so a reader that can seek over HTTP need only fetch the superblock,
the object headers and the chunks belonging to the datasets it wants.

    full archive, no Range      3500 GB   126 h at the measured 7.9 MB/s
    ranged, kept datasets only   149 GB     5.4 h

That is the difference between a five-day transfer and an afternoon, and it
is the single highest-leverage untested thing in the project.

THE TRAP, WHICH IS THIS PROJECT'S USUAL SHAPE
---------------------------------------------
`Accept-Ranges: bytes` in a HEAD response is a CLAIM, not evidence. A server
can advertise it and then answer a ranged GET with `200 OK` and the entire
body. Naive code slices the response, gets exactly the right bytes, returns
exactly the right array -- and transfers all 448 MB. Every test passes, the
data is correct, and the optimisation silently does not exist.

That is instance 6 again (an HTTP 200 that decoded as "the file is there").
So `probe_range_support` does not trust the header. It issues a real ranged
GET and requires ALL of:

  * status 206, not 200
  * a `Content-Range` header consistent with what was asked
  * exactly the requested number of bytes back
  * those bytes equal to the same slice fetched another way

and `RangedReader` counts the bytes actually transferred, so the saving is
measured rather than assumed. A reader reporting a 1.0x saving is the
signal that the server is lying.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

# 1 MiB, and raising it does nothing. Measured at fsspec's own _fetch layer
# -- where the HTTP range request is actually issued -- on a real 433 MB
# scan pulling the four IR channels and their LUTs:
#
#     block     fetches   wire MB   saving
#     256 KB       105      19.7     22.0x
#       1 MB       105      19.7     22.0x
#       4 MB       105      19.7     22.0x
#
# Identical, because h5py's reads already exceed the block size, so fsspec
# passes them through without over-fetching. The block size is not a lever
# here. Elapsed time varied 7.3-23.8 s across runs for identical work, so it
# is network variance, not a block-size effect -- which is why the earlier
# "bigger blocks are slower" reading was an artefact of reading timings as
# if they were measurements.
DEFAULT_BLOCK = 1 << 20


@dataclass
class RangeSupport:
    supported: bool
    reason: str
    advertises_header: bool = False
    status: int | None = None
    content_range: str | None = None
    bytes_returned: int | None = None
    content_length: int | None = None
    # Three outcomes, not two. "The server refused Range" and "the server
    # refused YOU" are different facts with different consequences, and the
    # first version collapsed them: an unauthenticated MOSDAC probe returned
    # 401 with a 59-byte body and was reported as "Range NOT usable", which
    # would have justified provisioning 3.4 TB of disk on an auth error.
    # That is instance 6 again -- reading an HTTP status as if it were data.
    #
    # Declared LAST on purpose: adding it third silently reassigned every
    # positional construction in this module (status became a Content-Range
    # string, and nothing raised).
    conclusive: bool = True

    def report(self) -> str:
        head = ("RANGE SUPPORTED" if self.supported else
                "INCONCLUSIVE" if not self.conclusive else "RANGE NOT USABLE")
        lines = [f"{head}: {self.reason}",
                 f"    advertises Accept-Ranges: {self.advertises_header}",
                 f"    ranged GET status: {self.status} "
                 f"(206 required; 200 means the whole body was sent)",
                 f"    Content-Range: {self.content_range}",
                 f"    bytes returned: {self.bytes_returned}"]
        if not self.conclusive:
            lines.append("    !! this says NOTHING about Range support. Do not "
                         "size storage or plan the archive on it -- re-run "
                         "authenticated.")
        if self.advertises_header and not self.supported and self.conclusive:
            lines.append("    !! the server ADVERTISES ranges and does not honour "
                         "them -- naive slicing would look correct and transfer "
                         "the whole file")
        return "\n".join(lines)


def probe_range_support(url, session=None, probe_len: int = 4096,
                        offset: int = 1024) -> RangeSupport:
    """Verify ranged GETs by doing one, not by reading a header."""
    import requests

    s = session or requests.Session()
    adv, clen = False, None
    try:
        h = s.head(url, timeout=60, allow_redirects=True)
        adv = h.headers.get("Accept-Ranges", "").lower() == "bytes"
        clen = int(h.headers.get("Content-Length", 0)) or None
    except Exception:
        pass

    end = offset + probe_len - 1
    r = s.get(url, headers={"Range": f"bytes={offset}-{end}"},
              timeout=120, stream=True)
    cr = r.headers.get("Content-Range")
    # Read only what was asked for and CLOSE. If the server ignores Range it
    # answers with the whole body, and `r.content` would quietly pull a
    # 448 MB scan to run a 4 KB probe -- turning a diagnostic into a transfer.
    try:
        body = next(r.iter_content(chunk_size=probe_len + 1), b"")
    finally:
        r.close()

    # Auth and redirect failures are NOT evidence about Range.
    if r.status_code in (401, 403, 407):
        return RangeSupport(
            False,
            f"{r.status_code} -- authentication required, so nothing was "
            f"learned about Range. Body was {len(body)} bytes "
            f"({body[:60]!r}).",
            adv, r.status_code, cr, len(body), clen, conclusive=False)
    if r.status_code >= 300 or len(body) < 512:
        return RangeSupport(
            False,
            f"status {r.status_code} with a {len(body)}-byte body -- that is "
            f"an error or login page, not a file. Nothing was learned about "
            f"Range.",
            adv, r.status_code, cr, len(body), clen, conclusive=False)
    if r.status_code != 206:
        return RangeSupport(False,
                            f"ranged GET returned {r.status_code}, not 206 -- "
                            f"the server sent the whole body",
                            adv, r.status_code, cr, len(body), clen)
    if not cr:
        return RangeSupport(False, "206 without a Content-Range header",
                            adv, r.status_code, cr, len(body), clen)
    if len(body) != probe_len:
        return RangeSupport(False,
                            f"asked for {probe_len} bytes, got {len(body)}",
                            adv, r.status_code, cr, len(body), clen)
    return RangeSupport(True, "verified by a real ranged GET",
                        adv, r.status_code, cr, len(body), clen)


class CountingFile(io.RawIOBase):
    """Wrap a seekable file-like and count the bytes actually read.

    The count is the whole point: it is the only way to tell a working
    ranged fetch from a server that quietly sent everything.
    """

    def __init__(self, inner):
        self.inner = inner
        self.bytes_read = 0
        self.n_reads = 0

    def read(self, size=-1):
        b = self.inner.read(size)
        self.bytes_read += len(b)
        self.n_reads += 1
        return b

    def readinto(self, b):
        data = self.inner.read(len(b))
        n = len(data)
        b[:n] = data
        self.bytes_read += n
        self.n_reads += 1
        return n

    def seek(self, off, whence=0):
        return self.inner.seek(off, whence)

    def tell(self):
        return self.inner.tell()

    def seekable(self):
        return True

    def readable(self):
        return True

    def close(self):
        try:
            self.inner.close()
        finally:
            super().close()


@dataclass
class RangedRead:
    datasets: dict = field(default_factory=dict)
    bytes_transferred: int = 0
    file_size: int | None = None
    n_requests: int = 0

    @property
    def saving(self) -> float:
        """file_size / bytes_transferred. 1.0 means no saving at all."""
        if not self.bytes_transferred or not self.file_size:
            return float("nan")
        return self.file_size / self.bytes_transferred

    def report(self) -> str:
        pct = (100 * self.bytes_transferred / self.file_size
               if self.file_size else float("nan"))
        lines = [f"fetched {len(self.datasets)} datasets, "
                 f"{self.bytes_transferred/1e6:.1f} MB transferred"]
        if self.file_size:
            lines.append(f"    whole file is {self.file_size/1e6:.1f} MB "
                         f"-> {pct:.1f}% fetched, {self.saving:.1f}x saving")
            if self.saving < 1.5:
                lines.append("    !! no meaningful saving -- the server is "
                             "probably answering ranged GETs with the full "
                             "body. Check probe_range_support().")
        return "\n".join(lines)


def open_ranged(url, session=None, block_size: int = DEFAULT_BLOCK):
    """A seekable, byte-counting file-like over HTTP. Caller closes it."""
    import fsspec

    kw = {}
    if session is not None:
        kw["headers"] = dict(session.headers)
    fs = fsspec.filesystem("http", block_size=block_size, **kw)
    return CountingFile(fs.open(url, "rb", block_size=block_size))


def read_datasets(url, names, session=None, block_size: int = DEFAULT_BLOCK,
                  file_size: int | None = None) -> RangedRead:
    """Read only `names` from a remote HDF5, over Range requests.

    Returns the arrays plus the bytes actually transferred, so the saving is
    a measurement. `names` are full HDF5 paths.
    """
    import h5py

    f = open_ranged(url, session, block_size)
    try:
        with h5py.File(f, "r") as h5:
            out = {}
            for n in names:
                if n not in h5:
                    continue
                out[n] = h5[n][...]
        return RangedRead(out, f.bytes_read, file_size, f.n_reads)
    finally:
        f.close()


def verify_against_local(url, local_path, names, session=None,
                         block_size: int = DEFAULT_BLOCK) -> dict:
    """Byte-identical check: ranged fetch vs the same file already on disk.

    This is the acceptance test. Equal shapes and a small tolerance would not
    do -- a wrong chunk index can return the right shape and plausible
    values, which is the failure this project keeps finding. Compares the
    raw bytes.
    """
    import os

    import h5py
    import numpy as np

    size = os.path.getsize(local_path)
    got = read_datasets(url, names, session, block_size, file_size=size)

    mismatches, checked = [], 0
    with h5py.File(local_path, "r") as ref:
        for n in names:
            if n not in ref:
                continue
            if n not in got.datasets:
                mismatches.append(f"{n}: missing from the ranged read")
                continue
            a = np.asarray(ref[n][...])
            b = np.asarray(got.datasets[n])
            checked += 1
            if a.shape != b.shape or a.dtype != b.dtype:
                mismatches.append(f"{n}: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
            elif a.tobytes() != b.tobytes():
                d = int(np.count_nonzero(np.asarray(a != b)))
                mismatches.append(f"{n}: {d} differing elements")
    return {
        "passed": not mismatches and checked > 0,
        "datasets_checked": checked,
        "mismatches": mismatches,
        "bytes_transferred": got.bytes_transferred,
        "file_size": size,
        "saving": got.saving,
        "n_requests": got.n_requests,
        "report": got.report(),
    }

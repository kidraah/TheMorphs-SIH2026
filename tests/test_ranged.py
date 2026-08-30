"""Ranged HDF5 fetch, tested against a real INSAT scan over a local server.

MOSDAC is down, but the question -- can we fetch 19 MB of datasets out of a
448 MB HDF5 over HTTP instead of the whole file -- does not depend on
MOSDAC. It depends on the HDF5 layout and on the server honouring Range. So
the layout half is answered NOW, on a real delivered file, served by a local
Range-capable server. What is left for MOSDAC is one probe.

The second server here LIES: it advertises `Accept-Ranges: bytes` and
answers every ranged GET with 200 and the whole body. Naive code returns
perfectly correct data from it while transferring everything, so the
optimisation would appear to work and would not exist. That server is the
point of the file.
"""
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from nowcast_data.ranged import (probe_range_support, read_datasets,
                                 verify_against_local)

SCAN_DIR = "/Users/evad/PycharmProjects/PythonProject1/data/insat/3RIMG_L1B_STD"
KEEP = ["IMG_TIR1", "IMG_WV", "Latitude", "Longitude",
        "Latitude_WV", "Longitude_WV", "IMG_TIR1_TEMP", "IMG_WV_TEMP"]


def _find_scan():
    for root, _, files in os.walk(SCAN_DIR):
        for f in sorted(files):
            if f.endswith(".h5"):
                return os.path.join(root, f)
    return None


SCAN = _find_scan()
needs_scan = pytest.mark.skipif(SCAN is None, reason="no delivered INSAT scan")


def _handler(path, honour_range: bool):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _body(self):
            size = os.path.getsize(path)
            rng = self.headers.get("Range")
            if rng and honour_range:
                units, _, spec = rng.partition("=")
                start, _, end = spec.partition("-")
                s = int(start)
                e = int(end) if end else size - 1
                e = min(e, size - 1)
                with open(path, "rb") as fh:
                    fh.seek(s)
                    data = fh.read(e - s + 1)
                self.send_response(206)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {s}-{e}/{size}")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return data
            # The liar: advertises ranges, ignores them, sends everything.
            with open(path, "rb") as fh:
                data = fh.read()
            self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return data

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(os.path.getsize(path)))
            self.end_headers()

        def do_GET(self):
            try:
                self.wfile.write(self._body())
            except (BrokenPipeError, ConnectionResetError):
                pass
    return H


class Server:
    def __init__(self, path, honour_range=True):
        self.httpd = HTTPServer(("127.0.0.1", 0), _handler(path, honour_range))
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/scan.h5"

    def __enter__(self):
        self.t = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


# ---------------------------------------------------------------------------

@needs_scan
def test_probe_accepts_a_server_that_really_honours_range():
    with Server(SCAN, True) as s:
        r = probe_range_support(s.url)
    assert r.supported, r.report()
    assert r.status == 206 and r.bytes_returned == 4096


@needs_scan
def test_probe_REJECTS_a_server_that_advertises_range_and_ignores_it():
    """THE test. This server sets Accept-Ranges: bytes and sends the whole
    body. A header check would pass it, naive slicing would return correct
    data, and the transfer saving would silently be 1.0x."""
    with Server(SCAN, False) as s:
        r = probe_range_support(s.url)
    assert not r.supported, r.report()
    assert r.advertises_header, "the liar must still advertise, or it is not the case we fear"
    assert r.status == 200
    assert "whole body" in r.reason


@needs_scan
def test_ranged_read_is_byte_identical_to_the_local_file():
    """Equal shapes and close values would not do: a wrong chunk index
    returns the right shape and plausible numbers."""
    with Server(SCAN, True) as s:
        res = verify_against_local(s.url, SCAN, KEEP)
    assert res["passed"], res["mismatches"]
    assert res["datasets_checked"] == len(KEEP)


@needs_scan
def test_ranged_read_actually_transfers_less():
    """The saving must be measured, not assumed."""
    with Server(SCAN, True) as s:
        res = verify_against_local(s.url, SCAN, KEEP)
    assert res["saving"] > 5.0, res["report"]
    assert res["bytes_transferred"] < 0.25 * res["file_size"]


@needs_scan
def test_the_lying_server_is_refused_by_the_reader_too():
    """Two independent defences, which is worth knowing rather than assuming.

    I expected this server to produce correct data with a 1.0x saving -- the
    silent version of the failure. It does not: fsspec detects that a seek
    past 0 returned the whole body and raises. So the liar is caught by the
    probe (by design) AND by the reader (by fsspec), and neither path can
    quietly transfer 448 MB while reporting success.

    The `saving` counter still earns its place: it is the only thing that
    would catch a server clever enough to satisfy fsspec while sending more
    than asked."""
    with Server(SCAN, False) as s:
        with pytest.raises(ValueError, match="range requests"):
            verify_against_local(s.url, SCAN, KEEP)


@needs_scan
def test_missing_datasets_are_reported_not_silently_skipped():
    with Server(SCAN, True) as s:
        res = verify_against_local(s.url, SCAN, KEEP + ["IMG_NOT_A_CHANNEL"])
    assert res["passed"], res["mismatches"]
    assert res["datasets_checked"] == len(KEEP)


class _Status:
    """A server that answers with a status and a short body, like an auth wall."""
    def __init__(self, code, body=b"Unauthorized"):
        self.code, self.body = code, body

    def __enter__(self):
        code, body = self.code, self.body

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_HEAD(self):
                self.send_response(code)
                self.end_headers()

            def do_GET(self):
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/x.h5"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *e):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_auth_failure_is_inconclusive_not_a_negative():
    """A 401 says the server refused YOU, not that it refused Range.

    The first version conflated them: an unauthenticated MOSDAC probe
    returned 401 with a 59-byte body and was reported as 'Range NOT usable',
    which would have justified provisioning 3.4 TB of disk on an auth error.
    """
    for code in (401, 403):
        with _Status(code) as s:
            r = probe_range_support(s.url)
        assert not r.conclusive, r.report()
        assert not r.supported
        assert "INCONCLUSIVE" in r.report()
        assert "says NOTHING about Range" in r.report()


def test_a_short_body_with_200_is_also_inconclusive():
    """A login page served with 200 is not a file."""
    with _Status(200, b"<html>Please log in</html>") as s:
        r = probe_range_support(s.url)
    assert not r.conclusive and "not a file" in r.reason


@needs_scan
def test_a_real_refusal_is_still_reported_as_conclusive():
    """The genuine negative must not be softened into 'inconclusive'."""
    with Server(SCAN, False) as s:
        r = probe_range_support(s.url)
    assert r.conclusive and not r.supported
    assert "RANGE NOT USABLE" in r.report()

"""The fetcher, against servers that fail the ways the real one failed.

The event pull produced one HTTP 500 and left seven .part files in 10.5
hours. Over a 5.3-day archive pull failures are certain, so each failure
mode gets a server here: flaky 500s, a truncating connection, a server that
will not resume, and one that returns a complete-looking HTML error page
with a 200 -- instance 6, which is the one a size check cannot catch.
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from nowcast_data.fetch import (COMPLETE, FAILED, FetchConfig, Fetcher, Item,
                                Manifest, VerificationError, sweep_orphaned_parts,
                                verify_hdf5)

DSETS = ("IMG_TIR1", "IMG_WV")


@pytest.fixture
def scan(tmp_path):
    import h5py
    p = tmp_path / "src.h5"
    rng = np.random.default_rng(0)
    with h5py.File(p, "w") as f:
        f.create_dataset("IMG_TIR1", data=rng.integers(0, 1024, (700, 700), "uint16"))
        f.create_dataset("IMG_WV", data=rng.integers(0, 1024, (350, 350), "uint16"))
        f.create_dataset("IMG_VIS", data=rng.integers(0, 1024, (1400, 1400), "uint16"))
    return p


def make_server(path, *, honour_range=True, fail_times=0, truncate_times=0,
                html_error=False):
    state = {"fails": fail_times, "truncs": truncate_times}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):
            pass

        def do_HEAD(self):
            self.send_response(200)
            if honour_range:
                self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(os.path.getsize(path)))
            self.end_headers()

        def do_GET(self):
            if state["fails"] > 0:
                state["fails"] -= 1
                self.send_error(500, "INTERNAL SERVER ERROR")
                return
            if html_error:
                body = (b"<html><body>Session expired. Please log in.</body>"
                        b"</html>" + b" " * (2 << 20))
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            size = os.path.getsize(path)
            rng = self.headers.get("Range")
            start = 0
            if rng and honour_range:
                spec = rng.split("=")[1]
                start = int(spec.split("-")[0])
                end_s = spec.split("-")[1]
                # Honour the END of the range too. The first version of this
                # server ignored it and returned everything from `start` --
                # and probe_range_support rejected it for returning more
                # bytes than were asked for, which is the probe working.
                end = int(end_s) if end_s else size - 1
                end = min(end, size - 1)
                with open(path, "rb") as fh:
                    fh.seek(start)
                    data = fh.read(end - start + 1)
                self.send_response(206)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                with open(path, "rb") as fh:
                    data = fh.read()
                self.send_response(200)
                if honour_range:
                    self.send_header("Accept-Ranges", "bytes")
            if state["truncs"] > 0:
                state["truncs"] -= 1
                data = data[:len(data) // 3]        # connection dies mid-body
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass
    return H


class Serve:
    def __init__(self, path, **kw):
        self.httpd = HTTPServer(("127.0.0.1", 0), make_server(path, **kw))
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/scan.h5"

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *e):
        self.httpd.shutdown()
        self.httpd.server_close()


def cfg(**kw):
    kw.setdefault("required_datasets", DSETS)
    kw.setdefault("backoff_base_s", 0.01)
    kw.setdefault("min_bytes", 1024)
    return FetchConfig(**kw)


# ---------------------------------------------------------------------------

def test_happy_path_writes_the_final_name_only_once_complete(scan, tmp_path):
    dest = tmp_path / "out" / "scan.h5"
    with Serve(scan) as s:
        f = Fetcher(tmp_path / "m.json", cfg(), log=lambda *a: None)
        f.fetch_all([Item(url=s.url, dest=str(dest), identifier="a")])
    assert dest.exists() and not dest.with_suffix(".h5.part").exists()
    assert f.manifest.items["a"].state == COMPLETE


def test_html_error_page_with_200_is_not_accepted(scan, tmp_path):
    """Instance 6. A 2 MB HTML page has a plausible size and passes any
    length check; it does not open as HDF5."""
    dest = tmp_path / "scan.h5"
    with Serve(scan, html_error=True) as s:
        f = Fetcher(tmp_path / "m.json", cfg(max_attempts=2), log=lambda *a: None)
        f.fetch_all([Item(url=s.url, dest=str(dest), identifier="a")])
    it = f.manifest.items["a"]
    assert it.state == FAILED
    assert "not HDF5" in it.error or "error page" in it.error
    assert not dest.exists(), "a bad file must never reach the final name"


def test_transient_500s_are_retried(scan, tmp_path):
    """The event pull hit exactly this."""
    dest = tmp_path / "scan.h5"
    with Serve(scan, fail_times=2) as s:
        f = Fetcher(tmp_path / "m.json", cfg(), log=lambda *a: None)
        f.fetch_all([Item(url=s.url, dest=str(dest), identifier="a")])
    it = f.manifest.items["a"]
    assert it.state == COMPLETE and it.attempts == 3


def test_gives_up_after_max_attempts_and_records_why(scan, tmp_path):
    dest = tmp_path / "scan.h5"
    with Serve(scan, fail_times=99) as s:
        f = Fetcher(tmp_path / "m.json", cfg(max_attempts=3), log=lambda *a: None)
        f.fetch_all([Item(url=s.url, dest=str(dest), identifier="a")])
    it = f.manifest.items["a"]
    assert it.state == FAILED and it.attempts == 3 and "500" in it.error
    assert f.manifest.failures()[0].key() == "a"


def test_truncated_body_is_caught_and_the_partial_is_discarded(scan, tmp_path):
    """A third of an HDF5 does not open. The partial must be deleted, not
    resumed onto -- appending to corrupt bytes gives a file of the right
    length that is wrong."""
    dest = tmp_path / "scan.h5"
    with Serve(scan, truncate_times=1) as s:
        f = Fetcher(tmp_path / "m.json", cfg(), log=lambda *a: None)
        f.fetch_all([Item(url=s.url, dest=str(dest), identifier="a")])
    it = f.manifest.items["a"]
    assert it.state == COMPLETE and it.attempts == 2
    import h5py
    with h5py.File(dest, "r") as g, h5py.File(scan, "r") as ref:
        assert g["IMG_TIR1"][...].tobytes() == ref["IMG_TIR1"][...].tobytes()


def test_resumes_from_a_partial_when_range_is_honoured(scan, tmp_path):
    dest = tmp_path / "scan.h5"
    part = tmp_path / "scan.h5.part"
    full = scan.read_bytes()
    part.write_bytes(full[:len(full) // 2])          # a real partial
    with Serve(scan) as s:
        f = Fetcher(tmp_path / "m.json", cfg(), log=lambda *a: None)
        f.fetch_all([Item(url=s.url, dest=str(dest), identifier="a")])
    assert dest.read_bytes() == full


def test_restarts_instead_of_appending_when_the_server_will_not_resume(scan, tmp_path):
    """Appending to a partial on a server that ignores Range concatenates the
    file onto its own tail: right length is impossible, but a naive
    implementation produces a corrupt file with no error."""
    dest = tmp_path / "scan.h5"
    part = tmp_path / "scan.h5.part"
    full = scan.read_bytes()
    part.write_bytes(full[:len(full) // 2])
    with Serve(scan, honour_range=False) as s:
        f = Fetcher(tmp_path / "m.json", cfg(), log=lambda *a: None)
        f.fetch_all([Item(url=s.url, dest=str(dest), identifier="a")])
    assert dest.read_bytes() == full


def test_restart_skips_completed_files(scan, tmp_path):
    dest = tmp_path / "scan.h5"
    mpath = tmp_path / "m.json"
    with Serve(scan) as s:
        Fetcher(mpath, cfg(), log=lambda *a: None).fetch_all(
            [Item(url=s.url, dest=str(dest), identifier="a")])
        # second run against a server that would fail if it were contacted
        with Serve(scan, fail_times=99) as bad:
            f2 = Fetcher(mpath, cfg(max_attempts=1), log=lambda *a: None)
            f2.fetch_all([Item(url=bad.url, dest=str(dest), identifier="a")])
    assert f2.manifest.items["a"].state == COMPLETE
    assert f2.manifest.items["a"].attempts == 1, "it should not have re-fetched"


def test_a_deleted_file_is_refetched_even_though_the_manifest_says_complete(scan, tmp_path):
    """Trusting the manifest over the filesystem is how a pull 'completes'
    with files missing."""
    dest = tmp_path / "scan.h5"
    mpath = tmp_path / "m.json"
    with Serve(scan) as s:
        Fetcher(mpath, cfg(), log=lambda *a: None).fetch_all(
            [Item(url=s.url, dest=str(dest), identifier="a")])
        dest.unlink()
        f2 = Fetcher(mpath, cfg(), log=lambda *a: None)
        f2.fetch_all([Item(url=s.url, dest=str(dest), identifier="a")])
    assert dest.exists() and f2.manifest.items["a"].state == COMPLETE


def test_dataset_mode_writes_only_the_kept_datasets(scan, tmp_path):
    """Range honoured -> fetch a subset and write a small HDF5. Everything
    downstream, including verification, is unchanged."""
    dest = tmp_path / "scan.h5"
    with Serve(scan) as s:
        f = Fetcher(tmp_path / "m.json", cfg(keep_datasets=DSETS),
                    log=lambda *a: None)
        f.fetch_all([Item(url=s.url, dest=str(dest), identifier="a")])
    it = f.manifest.items["a"]
    assert it.state == COMPLETE and it.mode == "datasets"
    import h5py
    with h5py.File(dest, "r") as g, h5py.File(scan, "r") as ref:
        assert set(g) == set(DSETS), "IMG_VIS must not be there"
        assert g["IMG_TIR1"][...].tobytes() == ref["IMG_TIR1"][...].tobytes()
        assert g.attrs["subset_of_full_scan"]
    assert dest.stat().st_size < scan.stat().st_size


def test_falls_back_to_whole_file_when_range_is_not_honoured(scan, tmp_path):
    """'Identical behaviour either way' -- same manifest, same verification,
    only the mode and the byte count differ."""
    dest = tmp_path / "scan.h5"
    with Serve(scan, honour_range=False) as s:
        f = Fetcher(tmp_path / "m.json", cfg(keep_datasets=DSETS),
                    log=lambda *a: None)
        f.fetch_all([Item(url=s.url, dest=str(dest), identifier="a")])
    it = f.manifest.items["a"]
    assert it.state == COMPLETE and it.mode == "whole"


def test_manifest_survives_a_kill_between_items(scan, tmp_path):
    dest1, dest2 = tmp_path / "a.h5", tmp_path / "b.h5"
    mpath = tmp_path / "m.json"
    with Serve(scan) as s:
        f = Fetcher(mpath, cfg(), log=lambda *a: None)
        f.fetch_all([Item(url=s.url, dest=str(dest1), identifier="a")])
        assert json.loads(mpath.read_text())["items"]["a"]["state"] == COMPLETE
        f2 = Fetcher(mpath, cfg(), log=lambda *a: None)
        f2.fetch_all([Item(url=s.url, dest=str(dest1), identifier="a"),
                      Item(url=s.url, dest=str(dest2), identifier="b")])
    assert Manifest(mpath).counts()[COMPLETE] == 2


def test_verify_rejects_a_file_missing_a_required_dataset(scan, tmp_path):
    with pytest.raises(VerificationError, match="missing datasets"):
        verify_hdf5(scan, ("IMG_TIR1", "IMG_NOPE"), min_bytes=1024)


def test_sweep_finds_orphaned_parts(tmp_path):
    p = tmp_path / "x.h5.part"
    p.write_bytes(b"0" * 10)
    os.utime(p, (0, 0))
    assert sweep_orphaned_parts(tmp_path) == [str(p)]
    assert sweep_orphaned_parts(tmp_path, older_than_s=1e12) == []


def test_report_shows_progress_and_failures(scan, tmp_path):
    with Serve(scan, fail_times=99) as s:
        f = Fetcher(tmp_path / "m.json", cfg(max_attempts=1), log=lambda *a: None)
        f.fetch_all([Item(url=s.url, dest=str(tmp_path / "a.h5"), identifier="a")])
    r = f.manifest.report()
    assert "1 failed" in r and "FAILED a" in r

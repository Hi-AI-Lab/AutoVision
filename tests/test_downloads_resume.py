# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""D11 resume/retry semantics of ``data.downloads._download_file``.

Every scenario runs against a local ``http.server`` bound to 127.0.0.1, so
no test touches the real network.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from autovision.data.downloads import DatasetFile, _download_file, _part_path
from autovision.errors import AutoVisionError


@dataclass
class _ScriptedResponse:
    """One scripted HTTP reply.

    ``serve`` is a truncation hook: the declared ``Content-Length`` covers
    the full body but only ``serve`` bytes hit the wire before the server
    closes the connection, exactly like a dropped weak-network transfer.
    """

    status: int
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    serve: int | None = None


class _ScriptedHandler(BaseHTTPRequestHandler):
    """Replays the server's scripted responses and logs every request."""

    protocol_version = "HTTP/1.0"

    def do_GET(self) -> None:  # noqa: N802 - http.server API spelling
        httpd = self.server
        assert isinstance(httpd, _ScriptedHTTPServer)
        httpd.requests.append(
            {key.lower(): value for key, value in self.headers.items()}
        )
        response = httpd.next_response()
        self.send_response(response.status)
        for key, value in response.headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(response.body)))
        self.end_headers()
        if response.serve is not None:
            self.wfile.write(response.body[: response.serve])
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(response.body)
        self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:
        """Silence per-request logging; assertions inspect recorded state."""


class _ScriptedHTTPServer(ThreadingHTTPServer):
    """Threading HTTP server carrying the script and the request log."""

    daemon_threads = True

    def __init__(self, script: list[_ScriptedResponse]) -> None:
        super().__init__(("127.0.0.1", 0), _ScriptedHandler)
        self.requests: list[dict[str, str]] = []
        self._script = list(script)
        self._lock = threading.Lock()

    @property
    def base_url(self) -> str:
        port = int(self.server_address[1])
        return f"http://127.0.0.1:{port}/payload.bin"

    def next_response(self) -> _ScriptedResponse:
        with self._lock:
            if not self._script:
                pytest.fail("test server received more requests than scripted")
            return self._script.pop(0)


@contextmanager
def _scripted_server(script: list[_ScriptedResponse]) -> Iterator[_ScriptedHTTPServer]:
    httpd = _ScriptedHTTPServer(list(script))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _sha256(payload: bytes) -> str:
    """Digest helper matching the registry's sha256 pinning."""
    return hashlib.sha256(payload).hexdigest()


def _record_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace the backoff sleep with a recorder so tests stay instant."""
    sleeps: list[float] = []
    monkeypatch.setattr(
        "autovision.data.downloads._sleep", lambda seconds: sleeps.append(seconds)
    )
    return sleeps


def test_truncated_transfer_resumes_with_206_and_passes_sha256(
    monkeypatch, tmp_path
):
    """A dropped transfer resumes from the *.part prefix via Range/206."""
    sleeps = _record_sleeps(monkeypatch)
    payload = bytes(range(256)) * 4
    cut = 700
    script = [
        _ScriptedResponse(200, body=payload, serve=cut),
        _ScriptedResponse(
            206,
            body=payload[cut:],
            headers={
                "Content-Range": f"bytes {cut}-{len(payload) - 1}/{len(payload)}"
            },
        ),
    ]
    dest = tmp_path / "payload.bin"
    with _scripted_server(script) as httpd:
        spec = DatasetFile(
            url=httpd.base_url, size_bytes=len(payload), sha256=_sha256(payload)
        )
        result = _download_file(spec, dest)
    assert result == dest
    assert dest.read_bytes() == payload
    assert not _part_path(dest).exists()
    assert len(httpd.requests) == 2
    assert httpd.requests[1]["range"] == f"bytes={cut}-"
    assert sleeps == [5.0]


def test_preexisting_part_resumes_on_the_first_request(monkeypatch, tmp_path):
    """A leftover *.part from an earlier run resumes via Range on request 1."""
    sleeps = _record_sleeps(monkeypatch)
    payload = b"payload resumed from a previous run"
    cut = 5
    script = [
        _ScriptedResponse(
            206,
            body=payload[cut:],
            headers={
                "Content-Range": f"bytes {cut}-{len(payload) - 1}/{len(payload)}"
            },
        ),
    ]
    dest = tmp_path / "payload.bin"
    _part_path(dest).write_bytes(payload[:cut])
    with _scripted_server(script) as httpd:
        spec = DatasetFile(
            url=httpd.base_url, size_bytes=len(payload), sha256=_sha256(payload)
        )
        result = _download_file(spec, dest)
    assert result == dest
    assert dest.read_bytes() == payload
    assert not _part_path(dest).exists()
    assert httpd.requests[0]["range"] == f"bytes={cut}-"
    assert sleeps == []


def test_server_ignoring_range_resets_the_partial_download(monkeypatch, tmp_path):
    """A Range-ignoring 200 answer restarts the download from zero."""
    sleeps = _record_sleeps(monkeypatch)
    payload = b"fresh complete payload"
    script = [
        _ScriptedResponse(200, body=payload),
        _ScriptedResponse(200, body=payload),
    ]
    dest = tmp_path / "payload.bin"
    _part_path(dest).write_bytes(b"stale prefix bytes")
    with _scripted_server(script) as httpd:
        spec = DatasetFile(url=httpd.base_url, sha256=_sha256(payload))
        _download_file(spec, dest)
    assert dest.read_bytes() == payload
    assert not _part_path(dest).exists()
    assert "range" in httpd.requests[0]
    assert "range" not in httpd.requests[1]
    assert sleeps == []


def test_416_resets_the_partial_download(monkeypatch, tmp_path):
    """A 416 answer to Range deletes the stale prefix and re-downloads."""
    sleeps = _record_sleeps(monkeypatch)
    payload = b"complete payload after a 416 reset"
    script = [
        _ScriptedResponse(416, body=b"range not satisfiable"),
        _ScriptedResponse(200, body=payload),
    ]
    dest = tmp_path / "payload.bin"
    _part_path(dest).write_bytes(b"stale prefix bytes")
    with _scripted_server(script) as httpd:
        spec = DatasetFile(url=httpd.base_url, sha256=_sha256(payload))
        _download_file(spec, dest)
    assert dest.read_bytes() == payload
    assert not _part_path(dest).exists()
    assert "range" in httpd.requests[0]
    assert "range" not in httpd.requests[1]
    assert sleeps == []


def test_mismatched_content_range_start_resets_the_download(monkeypatch, tmp_path):
    """A 206 whose Content-Range start disagrees with the prefix resets."""
    sleeps = _record_sleeps(monkeypatch)
    payload = b"payload after disagreeing content range"
    stale = b"stale prefix bytes"
    wrong_start = len(stale) - 3
    script = [
        _ScriptedResponse(
            206,
            body=payload,
            headers={
                "Content-Range": (
                    f"bytes {wrong_start}-{len(payload) - 1}/{len(payload)}"
                )
            },
        ),
        _ScriptedResponse(200, body=payload),
    ]
    dest = tmp_path / "payload.bin"
    _part_path(dest).write_bytes(stale)
    with _scripted_server(script) as httpd:
        spec = DatasetFile(url=httpd.base_url, sha256=_sha256(payload))
        _download_file(spec, dest)
    assert dest.read_bytes() == payload
    assert not _part_path(dest).exists()
    assert sleeps == []


def test_503_is_retried_after_injected_backoff_then_succeeds(monkeypatch, tmp_path):
    """A retryable 503 backs off once (injected no-op sleep) then succeeds."""
    sleeps = _record_sleeps(monkeypatch)
    payload = b"payload after a transient 503"
    script = [
        _ScriptedResponse(503, body=b"try again later"),
        _ScriptedResponse(200, body=payload),
    ]
    dest = tmp_path / "payload.bin"
    with _scripted_server(script) as httpd:
        spec = DatasetFile(url=httpd.base_url, sha256=_sha256(payload))
        _download_file(spec, dest)
    assert dest.read_bytes() == payload
    assert len(httpd.requests) == 2
    assert sleeps == [5.0]


def test_403_fails_fast_without_retry(monkeypatch, tmp_path):
    """A non-retryable 4xx raises immediately after a single request."""
    sleeps = _record_sleeps(monkeypatch)
    script = [
        _ScriptedResponse(403, body=b"forbidden"),
        _ScriptedResponse(403, body=b"forbidden"),
    ]
    dest = tmp_path / "payload.bin"
    with _scripted_server(script) as httpd:
        spec = DatasetFile(url=httpd.base_url, sha256=_sha256(b"unused"))
        with pytest.raises(AutoVisionError, match="Failed to download"):
            _download_file(spec, dest)
    assert len(httpd.requests) == 1
    assert sleeps == []
    assert not dest.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_insufficient_disk_space_is_refused_before_any_request(
    monkeypatch, tmp_path
):
    """Pinned sizes above 1 MiB pre-flight 1.5x free space and refuse."""
    sleeps = _record_sleeps(monkeypatch)
    size = 4 << 20

    class _TightDisk:
        """Fake disk_usage result just under the required headroom."""

        free = int(1.5 * size) - 1

    monkeypatch.setattr(
        "autovision.data.downloads.shutil.disk_usage", lambda _path: _TightDisk()
    )
    script: list[_ScriptedResponse] = []
    dest = tmp_path / "payload.bin"
    with _scripted_server(script) as httpd:
        spec = DatasetFile(url=httpd.base_url, size_bytes=size, sha256=_sha256(b"x"))
        with pytest.raises(AutoVisionError, match="Insufficient disk space"):
            _download_file(spec, dest)
    assert httpd.requests == []
    assert sleeps == []


def test_small_pinned_sizes_skip_the_disk_precheck(monkeypatch, tmp_path):
    """Sizes at or below the 1 MiB floor never consult disk usage."""
    payload = b"x" * (1 << 20)

    def _boom(_path):
        raise AssertionError("disk_usage must not be consulted")

    monkeypatch.setattr("autovision.data.downloads.shutil.disk_usage", _boom)
    script = [_ScriptedResponse(200, body=payload)]
    dest = tmp_path / "payload.bin"
    with _scripted_server(script) as httpd:
        spec = DatasetFile(
            url=httpd.base_url, size_bytes=len(payload), sha256=_sha256(payload)
        )
        _download_file(spec, dest)
    assert dest.read_bytes() == payload

# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Registry hardening batch 2 (H3-4i, H3-4j).

H3-4i: the downloadable cache used to read whole containers into memory,
write them directly to the final path, skip any cross-process mutual
exclusion, and re-hash the full file on every cache hit. It now streams
into a ``.part`` file published with ``os.replace``, serializes concurrent
resolves on a per-target FileLock, and memoizes the verified digest per
(path, size, mtime, expected sha).

H3-4j: a promotion blocked by its gates used to vanish silently (rmtree +
raise, audit log only recorded successes). Failed promotions are now
audited as ``promote_failed`` through the same hash-chain writer.
"""

from __future__ import annotations

import hashlib
import threading
import urllib.request
from pathlib import Path

import pytest
from filelock import FileLock

from autovision.registry import downloadable as dl
from autovision.registry.model_registry import ModelRegistry, RegistryError

# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def manifest():
    """Snapshot ASSET_MANIFEST and the digest memo around each test."""
    saved_manifest = dict(dl.ASSET_MANIFEST)
    dl._VERIFIED_FILES.clear()
    yield dl.ASSET_MANIFEST
    dl.ASSET_MANIFEST.clear()
    dl.ASSET_MANIFEST.update(saved_manifest)
    dl._VERIFIED_FILES.clear()


class _FakeResponse:
    """urlopen stand-in serving a fixed payload in ``read(n)`` chunks."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self, size: int = -1) -> bytes:
        if not self.payload:
            return b""
        chunk, self.payload = self.payload[:size], self.payload[size:]
        return chunk

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def _register_asset(name: str, payload: bytes, sha256: str | None = None) -> str:
    dl.register_downloadable(
        name,
        "file:///unused-test-endpoint",
        sha256 or hashlib.sha256(payload).hexdigest(),
    )
    return name


@pytest.fixture()
def chunked_urlopen(monkeypatch: pytest.MonkeyPatch):
    """Route urlopen to a chunked fake; returns a setter for the payload."""

    def _install(payload: bytes) -> list[int]:
        calls: list[int] = []

        def fake_urlopen(url: str, timeout: float | None = None) -> _FakeResponse:
            # 1 MiB read requests: the real _download asks for fixed chunks.
            calls.append(len(payload))
            return _FakeResponse(payload)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        return calls

    return _install


# ── H3-4i ────────────────────────────────────────────────────────────────


def test_download_streams_chunks_and_publishes_atomically(
    tmp_path: Path, manifest: dict, chunked_urlopen
) -> None:
    """H3-4i: the container lands byte-identical via chunked writes, with no
    ``.part`` residue at the final path."""
    payload = bytes(range(256)) * 8193  # ~2 MiB: several chunks
    name = _register_asset("h3-4i/streamed", payload)
    chunked_urlopen(payload)

    target = dl.resolve_downloadable(name, cache_dir=tmp_path)

    assert target.read_bytes() == payload
    assert target.name == "streamed.pt"
    assert not list(tmp_path.glob("**/*.part"))


def test_cache_hit_skips_full_file_rehash(
    tmp_path: Path, manifest: dict, chunked_urlopen, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H3-4i: a memoized (path, size, mtime) digest means the second resolve
    hashes nothing."""
    payload = b"cached-container" * 4096
    name = _register_asset("h3-4i/memo", payload)
    chunked_urlopen(payload)
    first = dl.resolve_downloadable(name, cache_dir=tmp_path)

    hashes: list[Path] = []
    real = dl.compute_file_sha256

    def counting_sha(path: Path, **kwargs: object) -> str:
        hashes.append(path)
        return real(path)

    monkeypatch.setattr(dl, "compute_file_sha256", counting_sha)
    second = dl.resolve_downloadable(name, cache_dir=tmp_path)

    assert second == first
    assert hashes == [], "cache hit re-read the whole file"


def test_sha256_mismatch_fails_closed_and_deletes(
    tmp_path: Path, manifest: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H3-4i: a mismatched download is deleted and raises — the cache must
    never hold unverified bytes under the real name."""
    name = _register_asset("h3-4i/mismatch", b"irrelevant", sha256="f" * 64)

    def fake_urlopen(url: str, timeout: float | None = None) -> _FakeResponse:
        return _FakeResponse(b"does-not-match")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(dl.DownloadableModelError, match="sha256 mismatch"):
        dl.resolve_downloadable(name, cache_dir=tmp_path)

    assert not (tmp_path / "h3-4i" / "mismatch.pt").exists()
    assert not list((tmp_path / "h3-4i").glob("*.part"))


def test_corrupted_cache_is_healed_by_redownload(
    tmp_path: Path, manifest: dict, chunked_urlopen
) -> None:
    """H3-4i: rewriting the file invalidates the digest memo (size/mtime in
    the key), so corruption still heals instead of trusting the cache."""
    payload = b"pristine-bytes" * 1024
    name = _register_asset("h3-4i/heal", payload)
    chunked_urlopen(payload)
    target = dl.resolve_downloadable(name, cache_dir=tmp_path)

    target.write_bytes(b"corrupted")
    chunked_urlopen(payload)
    healed = dl.resolve_downloadable(name, cache_dir=tmp_path)

    assert healed.read_bytes() == payload


def test_concurrent_resolve_serializes_on_file_lock(
    tmp_path: Path, manifest: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H3-4i: a resolve holding the target's lock excludes a second acquirer;
    the loser fails with an actionable timeout instead of racing the file.

    The competing holder lives in another thread: filelock singletons make
    same-thread re-acquisition reentrant (that is the registry's own nested
    locking story), so only cross-thread contention models a second process.
    """
    payload = b"locked-container"
    name = _register_asset("h3-4i/locked", payload)
    monkeypatch.setattr(dl, "_LOCK_TIMEOUT_SEC", 1.0)
    lock_path = tmp_path / "h3-4i" / ".locked.pt.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    held = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with FileLock(str(lock_path), timeout=5.0):
            held.set()
            release.wait(timeout=10.0)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    held.wait(timeout=5.0)
    try:
        with pytest.raises(dl.DownloadableModelError, match="Timed out"):
            dl.resolve_downloadable(name, cache_dir=tmp_path)
    finally:
        release.set()
        holder.join(timeout=10.0)


# ── H3-4j ────────────────────────────────────────────────────────────────


class _FailingGate:
    """Gate that rejects with a reason, like a real deployment gate."""

    name = "always-fail"

    def check(self, entry: object) -> list[str]:
        return ["intentional gate failure"]


class _ExplodingGate:
    """Gate that crashes: the boundary turns the exception into a reason."""

    name = "explodes"

    def check(self, entry: object) -> list[str]:
        raise TypeError("gate bug surfaced at the boundary")


def _register_entry(registry: ModelRegistry, tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"weights" * 64)
    registry.register(model, name="det", version="1.0.0")


@pytest.mark.parametrize(
    ("gates", "needle"),
    [
        pytest.param([_FailingGate()], "always-fail: intentional", id="gate-rejects"),
        pytest.param(
            [_ExplodingGate()], "explodes: TypeError", id="gate-raises"
        ),
    ],
)
def test_failed_promotion_is_audited_with_gate_reasons(
    tmp_path: Path, gates: list, needle: str
) -> None:
    """H3-4j: a blocked promotion lands a ``promote_failed`` audit entry
    carrying the gate name and reason summary, and the hash chain stays
    verifiable."""
    registry = ModelRegistry(tmp_path / "registry")
    _register_entry(registry, tmp_path)

    with pytest.raises(RegistryError, match="Deployment gates failed"):
        registry.promote(
            name="det", version="1.0.0", to_stage="staging", gates=gates
        )

    # audit_log() verifies the hash chain end-to-end; reaching a list at all
    # proves the promote_failed entry was chained correctly.
    log = registry.audit_log()
    failed = [record for record in log if record["action"] == "promote_failed"]
    assert len(failed) == 1
    record = failed[0]
    assert record["name"] == "det" and record["version"] == "1.0.0"
    assert record["stage"] == "staging"
    assert record["details"]["from_stage"] == "dev"
    assert any(needle in reason for reason in record["details"]["failures"])
    # The failed copy is gone; the source entry is untouched.
    assert not registry.exists(name="det", version="1.0.0", stage="staging")
    assert registry.exists(name="det", version="1.0.0", stage="dev")


def test_successful_promotion_records_only_the_success_event(
    tmp_path: Path,
) -> None:
    """H3-4j: passing gates produce the plain ``promote`` event — no
    promote_failed noise on the happy path."""
    from autovision.registry.deployment_gates import Sha256Gate

    registry = ModelRegistry(tmp_path / "registry")
    _register_entry(registry, tmp_path)
    registry.promote(
        name="det", version="1.0.0", to_stage="staging", gates=[Sha256Gate()]
    )

    actions = [record["action"] for record in registry.audit_log()]
    assert actions == ["register", "promote"]

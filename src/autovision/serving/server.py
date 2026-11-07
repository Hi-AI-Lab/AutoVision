# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""uvicorn launcher: TLS, multi-worker and single-process modes.

TLS and worker counts are transport concerns (not ASGI app concerns), so they
live here rather than in ``create_app``. Multi-worker mode requires a model
path because each worker process rebuilds the app from
``AUTOVISION_SERVE_CONFIG``; ready engine objects cannot be pickled into a
spawned worker.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from autovision.errors import AutoVisionError
from autovision.runtime.env import Env
from autovision.serving.app import create_app
from autovision.serving.worker import SERVE_CONFIG_ENV

__all__ = ["serve"]

# Binds outside this set are treated as network-exposed and require an API
# key (H2-1): the loopback aliases are the only hosts that cannot be reached
# from another machine.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise AutoVisionError(
        f"Serving option of type {type(value).__name__} is not JSON-serializable; "
        "multi-worker mode needs plain paths/scalars (use workers=1 for live objects)."
    )


def serve(
    model: Any = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    workers: int = 1,
    ssl_keyfile: str | Path | None = None,
    ssl_certfile: str | Path | None = None,
    ssl_ca_certs: str | Path | None = None,
    ssl_cert_reqs: int | None = None,
    timeout_graceful_shutdown: float | None = 30.0,
    limit_concurrency: int = 64,
    timeout_keep_alive: int = 30,
    **app_kwargs: Any,
) -> None:
    """Run the reference serving app through uvicorn.

    ``workers > 1`` requires ``model`` to be a path (each worker rebuilds the
    app); ``variants=``/engine objects are single-process only. Provide both
    ``ssl_keyfile`` and ``ssl_certfile`` to enable TLS.

    Authentication is fail-closed for non-loopback binds (H2-1): serving on a
    non-loopback ``host`` without ``api_key=`` and without
    ``AUTOVISION_SERVING_API_KEY`` in the environment refuses to start; set
    ``AUTOVISION_SERVE_ALLOW_INSECURE=1`` to accept an open endpoint
    explicitly. ``api_key=`` is rejected for ``workers > 1`` (H2-4): the
    multi-worker path serializes app options into an environment variable
    that same-UID processes can read via ``/proc``, so the key must come from
    ``AUTOVISION_SERVING_API_KEY`` instead (worker subprocesses inherit it).

    ``limit_concurrency`` and ``timeout_keep_alive`` bound uvicorn's
    connection resources (production should layer gateway-side quotas on
    top). ``timeout_graceful_shutdown`` bounds the SIGTERM drain window and
    defaults to 30s; pass ``None`` explicitly to wait indefinitely (uvicorn's
    own default).
    """
    import uvicorn

    if timeout_graceful_shutdown is not None and timeout_graceful_shutdown <= 0:
        raise AutoVisionError(
            "timeout_graceful_shutdown must be > 0 or None, got "
            f"{timeout_graceful_shutdown}."
        )
    if limit_concurrency < 1:
        raise AutoVisionError(
            f"limit_concurrency must be >= 1, got {limit_concurrency}."
        )
    if timeout_keep_alive < 1:
        raise AutoVisionError(
            f"timeout_keep_alive must be >= 1, got {timeout_keep_alive}."
        )
    if workers < 1:
        raise AutoVisionError(f"workers must be >= 1, got {workers}.")
    if (ssl_keyfile is None) != (ssl_certfile is None):
        raise AutoVisionError(
            "TLS needs both ssl_keyfile= and ssl_certfile= (one without the "
            "other cannot start a server)."
        )
    if (ssl_ca_certs is not None or ssl_cert_reqs is not None) and ssl_keyfile is None:
        raise AutoVisionError(
            "ssl_ca_certs/ssl_cert_reqs require TLS (ssl_keyfile + ssl_certfile)."
        )
    # H2-1: enforce the fail-closed posture here rather than in create_app —
    # create_app stays a library entry point (in-process tests), while serve()
    # is the actual network-exposed launch path. An explicitly passed api_key
    # satisfies the guard; workers>1 + api_key is rejected further below.
    key_present = (
        app_kwargs.get("api_key") is not None
        or Env.SERVING_API_KEY.value is not None
    )
    if (
        host not in _LOOPBACK_HOSTS
        and not key_present
        and not Env.SERVE_ALLOW_INSECURE.value
    ):
        raise AutoVisionError(
            "Refusing to bind a non-loopback address without authentication "
            "(fail-closed). Provide api_key=, or set AUTOVISION_SERVING_API_KEY "
            "in the environment, or set AUTOVISION_SERVE_ALLOW_INSECURE=1 to "
            "accept an open endpoint explicitly."
        )
    tls_kwargs: dict[str, Any] = {}
    if ssl_keyfile is not None:
        tls_kwargs["ssl_keyfile"] = str(ssl_keyfile)
        tls_kwargs["ssl_certfile"] = str(ssl_certfile)
    if ssl_ca_certs is not None:
        tls_kwargs["ssl_ca_certs"] = str(ssl_ca_certs)
    if ssl_cert_reqs is not None:
        tls_kwargs["ssl_cert_reqs"] = int(ssl_cert_reqs)
    # H2-2a/H2-5e: connection limits always apply (safe defaults); the
    # graceful-shutdown timeout is omitted only when the caller asks for the
    # unbounded wait via None.
    server_kwargs: dict[str, Any] = {
        "limit_concurrency": limit_concurrency,
        "timeout_keep_alive": timeout_keep_alive,
    }
    if timeout_graceful_shutdown is not None:
        server_kwargs["timeout_graceful_shutdown"] = timeout_graceful_shutdown

    if workers == 1:
        app = create_app(model, **app_kwargs)
        uvicorn.run(app, host=host, port=port, **server_kwargs, **tls_kwargs)
        return

    if "variants" in app_kwargs:
        raise AutoVisionError(
            "variants= is not supported with workers>1; run one model per "
            "worker deployment or route variants at the gateway."
        )
    if model is None or not isinstance(model, (str, Path)):
        raise AutoVisionError(
            "workers>1 requires model=<path>; live engine/model objects cannot "
            "be rebuilt inside worker processes."
        )
    # H2-4: the config payload lands in an environment variable (readable by
    # every same-UID process via /proc/<pid>/environ), so the credential must
    # travel through AUTOVISION_SERVING_API_KEY, which subprocesses inherit.
    if app_kwargs.get("api_key") is not None:
        raise AutoVisionError(
            "api_key= cannot be used with workers>1: the key would be written "
            f"into {SERVE_CONFIG_ENV} and leak via /proc/<pid>/environ. Set "
            "AUTOVISION_SERVING_API_KEY instead; worker subprocesses inherit "
            "the environment."
        )
    config = {"model": str(model), **app_kwargs}
    # SERVE-07: pre-validate the config in the PARENT process — the old
    # shape exploded inside every worker as a uvicorn restart loop with no
    # clean CLI error. Unknown keys surface here as a named error too.
    from autovision.serving.app import validate_serve_config

    try:
        # ``model`` is the launch payload, not a serving knob — validate the
        # rest so a misspelled option still surfaces as a TypeError here.
        validate_serve_config(
            **{key: value for key, value in config.items() if key != "model"}
        )
    except TypeError as error:
        raise AutoVisionError(
            f"Unknown or misspelled serving option: {error}"
        ) from error
    except ValueError as error:
        # Review P2-1: a numeric knob carrying a non-numeric string raised
        # a bare ValueError that bypassed the named-error treatment.
        raise AutoVisionError(f"Invalid serving option: {error}") from error
    try:
        payload = json.dumps(_jsonable(config), sort_keys=True)
    except TypeError as error:
        raise AutoVisionError(f"Cannot serialize serving config: {error}") from error
    os.environ[SERVE_CONFIG_ENV] = payload
    try:
        uvicorn.run(
            "autovision.serving.worker:create_app_from_env",
            factory=True,
            host=host,
            port=port,
            workers=workers,
            **server_kwargs,
            **tls_kwargs,
        )
    finally:
        # SERVE-09: the env payload outlives uvicorn in this process — a
        # later create_app_from_env() in the same interpreter would pick up
        # the stale config.
        os.environ.pop(SERVE_CONFIG_ENV, None)

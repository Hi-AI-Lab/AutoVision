#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Plugin protocol smoke test (``plugins`` subsystem).

Exercises the plugin contract the registry relies on for
``model_source`` / ``dataset`` / ``metric`` / ``loss`` / ``task`` /
``postprocess`` extensions: register a plugin, enumerate plugins by kind,
and retrieve one. Pure in-memory — no assets or network required.

    python examples/tasks/plugin_demo.py
"""

from __future__ import annotations

import argparse

from autovision.plugins import (
    PluginInfo,
    get_plugin,
    list_plugins,
    register_plugin,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    def _demo_metric() -> str:
        return "demo"

    info = PluginInfo(
        name="demo-metric",
        version="0.1",
        kind="metric",
        factory=_demo_metric,
        description="example metric plugin for the smoke test",
    )
    registered = register_plugin(info)
    print("registered:", registered.name, "| kind =", registered.kind)

    names = [plugin.name for plugin in list_plugins(kind="metric")]
    print("metric plugins:", names)

    got = get_plugin("metric", "demo-metric")
    print("get_plugin ->", got.name, "| factory() =", got.factory())

    print("\nregistered plugins by kind:")
    for kind in ("model_source", "dataset", "metric", "loss", "task", "postprocess"):
        print(f"  {kind}: {[plugin.name for plugin in list_plugins(kind=kind)]}")


if __name__ == "__main__":
    main()

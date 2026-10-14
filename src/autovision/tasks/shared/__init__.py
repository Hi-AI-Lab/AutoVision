# Copyright (c) 2026 Autovision contributors. Apache-2.0.
"""Cross-task training infrastructure shared by all task families.

Modules here are task-agnostic (schedules, EMA, deployment tiling) and
must not import from any single task package.
"""

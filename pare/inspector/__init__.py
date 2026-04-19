"""Trajectory Inspector — the Pare 2.0 MVP.

CLI: ``pare inspect`` (see :mod:`pare.inspector.cli`).

Pipeline (W1-W2 deliverables):

    JSONL / Langfuse trace
        → loader.py        (TrajectoryRecord)
        → annotator.py     (AnnotatedTrajectory: Liu tags + recovery labels)
        → differ.py        (align two trajectories, find DivergencePoint)
        → html_renderer.py (Jinja2 → standalone HTML report)
"""

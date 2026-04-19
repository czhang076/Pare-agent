"""Render an inspection report to a single self-contained HTML file.

R0 scaffold — types only.

Hard constraint: no JS framework. Server-rendered Jinja2 + native
``<details>`` for collapsibles + a tiny inline CSS file. The report
should open and read correctly when served from ``file://`` with no
network. This is a one-person 4-week MVP; do not reach for React.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pare.inspector.annotator import AnnotatedTrajectory
    from pare.inspector.differ import AlignedStep, DivergencePoint


@dataclass(frozen=True, slots=True)
class InspectionReport:
    """View-model for the HTML template. One trajectory or two."""

    title: str                                          # e.g. "sympy__sympy-11618"
    primary: "AnnotatedTrajectory"                      # always present
    comparison: "AnnotatedTrajectory | None" = None     # set in --diff mode
    aligned: "list[AlignedStep] | None" = None          # set in --diff mode
    divergence: "DivergencePoint | None" = None         # set in --diff mode


def render(report: InspectionReport) -> str:
    """Render ``report`` to a complete HTML document string.

    Inlines the CSS so the output file is portable. Loads the Jinja2
    template from ``pare/inspector/templates/report.html.j2`` via
    ``importlib.resources`` so the package works when zipped.
    """
    raise NotImplementedError("W2 Day 1-3")

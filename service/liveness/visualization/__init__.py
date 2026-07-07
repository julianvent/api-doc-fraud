"""Debug + explainability overlays and per-request HTML reports."""

from service.liveness.visualization.html_report import render as render_html_report
from service.liveness.visualization.html_report import write as write_html_report
from service.liveness.visualization.overlays import annotate

__all__ = ["annotate", "render_html_report", "write_html_report"]

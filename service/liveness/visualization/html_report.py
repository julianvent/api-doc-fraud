"""Self-contained HTML report for a single LivenessReport.

A single-file HTML page referencing the annotated image and face crop
via relative `<img src="...">` (works from the artifact directory).
Dependency-free: a string builder, no templating engine.
"""

from __future__ import annotations

import html
from pathlib import Path

from service.liveness.domain.report import LivenessReport
from service.liveness.domain.verdict import Verdict


_VERDICT_CSS: dict[Verdict, str] = {
    Verdict.ACCEPT: "verdict-accept",
    Verdict.REVIEW: "verdict-review",
    Verdict.HARD_REJECT: "verdict-reject",
}


_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 32px;
       background: #0f1115; color: #e6e6e6; }
h1 { margin-top: 0; }
.container { max-width: 1100px; margin: 0 auto; }
.verdict { font-size: 28px; font-weight: 700; padding: 8px 16px; border-radius: 8px;
           display: inline-block; }
.verdict-accept { background: #1f6f3a; color: #fff; }
.verdict-review { background: #b07b15; color: #fff; }
.verdict-reject { background: #a32626; color: #fff; }
.score { font-size: 18px; opacity: 0.8; margin-left: 12px; }
.grid { display: grid; grid-template-columns: 2fr 1fr; gap: 24px; margin-top: 24px; }
.card { background: #1a1d24; border: 1px solid #262a33; border-radius: 10px; padding: 16px; }
.card img { width: 100%; border-radius: 6px; display: block; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { padding: 8px 6px; text-align: left; border-bottom: 1px solid #262a33; }
th { color: #9ba0aa; font-weight: 500; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px;
       background: #262a33; margin-right: 4px; }
.status-OK { color: #6dd47e; }
.status-FAILED { color: #ff6b6b; }
.status-SKIPPED { color: #9ba0aa; }
.meta { color: #9ba0aa; font-size: 13px; margin-top: 8px; }
.section-title { font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em;
                 color: #9ba0aa; margin: 0 0 12px 0; }
ul.reasons { padding-left: 18px; margin: 0; }
ul.reasons li { padding: 2px 0; }
.timing td { font-family: ui-monospace, Menlo, monospace; }
"""


def render(report: LivenessReport, annotated_filename: str | None = None) -> str:
    """Return an HTML document string for the given report.

    `annotated_filename` is the basename of the annotated image inside
    the same directory as the HTML. Defaults to "annotated.jpg".
    """
    annotated_filename = annotated_filename or "annotated.jpg"
    crop_filename = "crop.jpg" if report.face_crop and report.face_crop.crop_path else None

    verdict_css = _VERDICT_CSS.get(report.verdict, "verdict-review")
    reasons_html = (
        "<ul class='reasons'>"
        + "".join(f"<li>{html.escape(r)}</li>" for r in report.reasons)
        + "</ul>"
        if report.reasons
        else "<p class='meta'>no reasons</p>"
    )

    detector_rows = []
    for name, det in report.detectors.items():
        detector_rows.append(
            f"<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{det.score:.3f}</td>"
            f"<td><span class='status-{det.status.value}'>{det.status.value}</span></td>"
            f"<td>{det.latency_ms:.1f} ms</td>"
            f"<td>{html.escape('; '.join(det.reasons)) or '—'}</td>"
            f"</tr>"
        )
    detector_html = (
        "<table><tr><th>name</th><th>score</th><th>status</th>"
        "<th>latency</th><th>reasons</th></tr>"
        + "".join(detector_rows)
        + "</table>"
        if detector_rows
        else "<p class='meta'>no detectors ran (quality gate or face not detected)</p>"
    )

    timing_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v:.2f} ms</td></tr>"
        for k, v in sorted(report.timing_ms.items())
    )
    timing_html = (
        "<table class='timing'>" + timing_rows + "</table>"
        if timing_rows
        else "<p class='meta'>no timing recorded</p>"
    )

    quality_html = ""
    if report.face_crop is not None:
        q = report.face_crop.quality
        quality_html = (
            f"<table>"
            f"<tr><th>blur</th><td>{q.blur_score:.3f}</td></tr>"
            f"<tr><th>brightness</th><td>{q.brightness_score:.3f}</td></tr>"
            f"<tr><th>occlusion</th><td>{q.occlusion_score:.3f}</td></tr>"
            f"<tr><th>acceptable</th><td>{q.is_acceptable}</td></tr>"
            f"</table>"
        )

    crop_block = (
        f"<div class='card'><div class='section-title'>face crop</div>"
        f"<img src='{html.escape(crop_filename)}'/>{quality_html}</div>"
        if crop_filename
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>Liveness report {html.escape(report.request_id[:8])}</title>
<style>{_CSS}</style>
</head>
<body>
<div class='container'>
  <h1>Liveness Detection Report</h1>
  <div>
    <span class='verdict {verdict_css}'>{report.verdict.value}</span>
    <span class='score'>score = {report.score:.3f}</span>
  </div>
  <p class='meta'>
    request_id: {html.escape(report.request_id)}
    &nbsp;·&nbsp; timestamp: {html.escape(report.timestamp.isoformat())}
    <br>
    module {html.escape(report.module_version)}
    &nbsp;·&nbsp; thresholds {html.escape(report.thresholds_version)}
  </p>

  <div class='grid'>
    <div class='card'>
      <div class='section-title'>annotated input</div>
      <img src='{html.escape(annotated_filename)}'/>
    </div>
    {crop_block}
  </div>

  <div class='grid'>
    <div class='card'>
      <div class='section-title'>detectors</div>
      {detector_html}
    </div>
    <div class='card'>
      <div class='section-title'>reasons</div>
      {reasons_html}
    </div>
  </div>

  <div class='card' style='margin-top:24px'>
    <div class='section-title'>timing</div>
    {timing_html}
  </div>
</div>
</body>
</html>
"""


def write(report: LivenessReport, directory: Path | str,
          annotated_filename: str = "annotated.jpg") -> Path:
    """Write the HTML report next to the artifacts. Returns the path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "report.html"
    path.write_text(render(report, annotated_filename), encoding="utf-8")
    return path

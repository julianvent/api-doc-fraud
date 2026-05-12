import json
import re

from service.ocr.agent.base import LLMBackend
from service.ocr.models import Config, MRZResult, PipelineOutput, TextLine
from service.ocr.normalizer import normalize_fields


def _load_fields(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _mrz_to_dict(mrz: MRZResult) -> dict:
    raw = {
        "surname"         : mrz.surname,
        "given_names"     : mrz.given_names,
        "country"         : mrz.country,
        "date_of_birth"   : mrz.birth_date,
        "date_of_expiry"  : mrz.expiry_date,
        "document_number" : mrz.number,
        "sex"             : mrz.sex
    }
    return normalize_fields({k: v for k, v in raw.items() if v})


def _get_missing_fields(all_fields: list, resolved: dict) -> list:
    return [f for f in all_fields if f not in resolved]


def _get_image_dimensions(lines: list[TextLine]) -> tuple[float, float]:
    max_x, max_y = 1.0, 1.0
    for line in lines:
        if line.bbox is None or len(line.bbox) == 0:
            continue
        xs = [pt[0] for pt in line.bbox]
        ys = [pt[1] for pt in line.bbox]
        max_x = max(max_x, max(xs))
        max_y = max(max_y, max(ys))
    return max_x, max_y


def _build_spatial_layout(lines: list[TextLine]) -> str:
    if not lines:
        return ""

    img_w, img_h = _get_image_dimensions(lines)

    def y_center(line: TextLine) -> float:
        if line.bbox is None or len(line.bbox) == 0:
            return 0.0
        ys = [pt[1] for pt in line.bbox]
        return round(((min(ys) + max(ys)) / 2) / img_h, 3)

    def x_start(line: TextLine) -> float:
        if line.bbox is None or len(line.bbox) == 0:
            return 0.0
        xs = [pt[0] for pt in line.bbox]
        return round(min(xs) / img_w, 3)

    sorted_lines = sorted(lines, key=lambda l: (y_center(l), x_start(l)))

    return "\n".join(
        f"[y={y_center(l):.3f} x={x_start(l):.3f}] {l.text}"
        for l in sorted_lines
    )


def _build_label_hints_section(label_hints: dict, missing_fields: list) -> str:
    if not label_hints:
        return ""
    relevant = {k: v for k, v in label_hints.items() if k in missing_fields}
    if not relevant:
        return ""
    lines = ["## Label hints for field extraction"]
    lines.append("Each field may appear under one of these labels in the document:")
    for field, hints in relevant.items():
        lines.append(f"  {field}: look for labels like {', '.join(hints)}")
    return "\n".join(lines)


def _build_prompt(output: PipelineOutput,
                  mrz_fields: dict,
                  mrz_valid: bool | None) -> str:
    spatial_layout = _build_spatial_layout(output.english_lines)

    mrz_section = ""
    if mrz_fields:
        mrz_status  = "VERIFIED (checksum passed)" if mrz_valid else "UNVERIFIED (checksum failed — possible tampering)"
        mrz_section = f"""
## MRZ reference — {mrz_status}
These values come from the Machine Readable Zone and are provided ONLY for inconsistency detection.
DO NOT copy them into your extraction. Extract every field independently from the document layout, then compare against MRZ in Step 3.
{json.dumps(mrz_fields, ensure_ascii=False, indent=2)}
"""

    return f"""You are a strict document validation agent for official identity documents.

## Critical extraction rules
- A field value is the text immediately to the right of its label (same y) OR the single line directly below it (next y, similar x).
- NEVER concatenate multiple lines into a single field value.
- NEVER include another field's label or value inside a field value.
- If a value seems too long (more than 5 words for names, more than 12 chars for dates/numbers), it is likely wrong — re-check the coordinates.
- Extract values exactly as they appear — do not interpret, translate, or reformat.

## Document layout
Each line has normalized spatial coordinates [y=row x=column] (0.0 to 1.0).
Lines with similar y values are on the same row.
A label and its value are typically on the same row (same y, different x) or the value is on the next row directly below the label (slightly higher y, similar x).

{spatial_layout}
{mrz_section}

## Step 1 — Identify document type
Infer the document type from the layout content (e.g. "visa", "passport", "national_id", "driver_license"). Use a short snake_case identifier. If you cannot determine it, use "unknown".

## Step 2 — Discover every label and extract its value
Scan the layout and find every line that acts as a label (the descriptive text that introduces a field — e.g. "Surname", "Date of Issue", "Passport No", "Visa Type").
For each label:
1. Derive a snake_case field key from the label text (e.g. "Date of Expiry (DD/MM/YYYY)" → "date_of_expiry", "Passport No." → "passport_no", "उपमा और नाम /Surname and Given Name" → "surname_and_given_name"). Use only ASCII letters, digits and underscores in the key.
2. Find the value: the SAME ROW (same y ±0.02) to the right, OR the NEXT ROW (y +0.02 to +0.06) at similar x.
3. Take ONLY that single line as the value. If you cannot determine a value, set it to null.
4. Do NOT invent fields that have no clear label in the layout.
5. Do NOT copy values from the MRZ reference — extract every field independently from the document layout.
6. Also produce an `extractions` array with the EXACT layout text used for each pair (this is needed so the result can be audited visually):
   - `key`        : the same snake_case key
   - `label_text` : the verbatim text of the label line as it appears in the layout
   - `value_text` : the verbatim text of the value line (empty string "" if the value is null)

## Step 3 — Detect inconsistencies (this is where MRZ is used)
An inconsistency is ONLY:
- date_of_issue is after date_of_expiry
- A visually extracted field directly contradicts the same MRZ field
- MRZ checksum failed (flagged above if applicable)

Not inconsistencies:
- Future dates (normal for date_of_expiry)
- Missing fields
- Anything not directly verifiable from the document

## Step 4 — Verdict
genuine    → all extracted fields consistent, MRZ valid or absent
suspicious → at least one confirmed inconsistency or MRZ checksum failed

Return ONLY raw JSON, no explanation, no markdown, no preamble:
{{
    "document_type": "<snake_case type>",
    "fields": {{
        "<derived_snake_case_key>": "<single line value as found or null>"
    }},
    "extractions": [
        {{
            "key"       : "<derived_snake_case_key>",
            "label_text": "<verbatim label line as in layout>",
            "value_text": "<verbatim value line as in layout, or empty string>"
        }}
    ],
    "inconsistencies": [
        {{
            "field"      : "<field key>",
            "description": "<exact contradiction between two specific values>"
        }}
    ],
    "confidence": "<high | medium | low>",
    "verdict"   : "<genuine | suspicious>",
    "notes"     : "<critical observation only, or null>"
}}"""


def _parse_response(raw: str) -> dict:
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    return json.loads(clean)


def fill_missing_fields(lines: list[TextLine],
                         backend: LLMBackend,
                         field_keys: list[str]) -> dict[str, str | None]:
    if not field_keys or not lines:
        return {}

    spatial = _build_spatial_layout(lines)
    keys_json = json.dumps(field_keys, ensure_ascii=False)

    prompt = f"""You are extracting specific fields from an identity document.

## Document layout (normalized coordinates)
{spatial}

## Fields to extract
{keys_json}

## Instructions
- Interpret each field key as a hint of what to look for (e.g. "passport_no" = passport number).
- A value is the text adjacent (same row to the right, or next row directly below) to a label that semantically matches the key.
- Take ONLY a single line as the value.
- If a field cannot be found, set its value to null.
- Do NOT include the label text in the value.

Return ONLY raw JSON, no explanation, no markdown:
{{
    "<field_key>": "<value or null>"
}}"""

    try:
        raw = backend.complete(prompt)
        parsed = _parse_response(raw)
        return {k: parsed.get(k) for k in field_keys}
    except Exception:
        return {}


def analyze(output: PipelineOutput,
            config: Config,
            backend: LLMBackend) -> dict:

    mrz        = output.mrz
    mrz_fields = _mrz_to_dict(mrz) if mrz else {}
    mrz_valid  = mrz.valid if mrz else None

    prompt = _build_prompt(output, mrz_fields, mrz_valid)

    raw = backend.complete(prompt)

    try:
        agent_result = _parse_response(raw)
    except Exception:
        agent_result = {"error": "agent response could not be parsed", "raw": raw}

    doc_type     = agent_result.get("document_type", "default")
    agent_fields = normalize_fields(agent_result.get("fields", {}))
    extractions  = agent_result.get("extractions", []) or []

    mrz_output = None
    if output.mrz_verified:
        mrz_output = {
            "valid" : True,
            "source": "mrz_verified",
            "fields": mrz_fields
        }
    elif output.mrz_unverified:
        mrz_output = {
            "valid"  : False,
            "source" : "mrz_unverified",
            "fields" : mrz_fields,
            "warning": "MRZ checksum failed — fields may be unreliable"
        }

    return {
        "mrz_fields": mrz_output,

        "agent_fields": {
            "source"         : backend.__class__.__name__,
            "document_type"  : doc_type,
            "fields"         : agent_fields,
            "inconsistencies": agent_result.get("inconsistencies", []),
            "confidence"     : agent_result.get("confidence"),
            "verdict"        : agent_result.get("verdict"),
            "notes"          : agent_result.get("notes")
        },

        "result": {
            "document_type"  : doc_type,
            "fields"         : agent_fields,
            "extractions"    : extractions,
            "inconsistencies": agent_result.get("inconsistencies", []),
            "verdict"        : agent_result.get("verdict"),
            "confidence"     : agent_result.get("confidence"),
            "mrz_valid"      : mrz_valid,
            "source"         : output.source,
            "confidence_avg" : output.confidence_avg
        }
    }
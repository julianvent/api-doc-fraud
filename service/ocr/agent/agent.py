import json
import re

from service.ocr.agent.base import LLMBackend
from service.ocr.models import Config, MRZResult, PipelineOutput, TextLine
from service.ocr.normalizer import normalize_fields


def _load_fields(path: str) -> dict:
    try:
        with open(path, "r") as f:
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
                  fields_map: dict,
                  missing_fields: list,
                  mrz_fields: dict,
                  mrz_valid: bool | None,
                  label_hints: dict) -> str:    # en lugar de pasar el JSON completo con indent=2
    schemas_json = json.dumps(
        {k: v["fields"] if isinstance(v, dict) else v for k, v in fields_map.items()},
        ensure_ascii=False,
        separators=(',', ':')  # sin espacios — menos tokens
    )
    missing_str    = json.dumps(missing_fields, ensure_ascii=False)
    spatial_layout = _build_spatial_layout(output.english_lines)
    hints_section  = _build_label_hints_section(label_hints, missing_fields)

    mrz_section = ""
    if mrz_fields:
        mrz_status = "VERIFIED (checksum passed)" if mrz_valid else "UNVERIFIED (checksum failed — possible tampering)"
        mrz_section = f"""
## MRZ extracted fields — {mrz_status}
These fields were extracted from the Machine Readable Zone and must NOT be re-extracted.
{"Use them as ground truth for cross-validation." if mrz_valid else "Treat with caution — checksum failure may indicate document tampering."}
{json.dumps(mrz_fields, ensure_ascii=False, indent=2)}
"""

    return f"""You are a strict document validation agent specialized in official identity documents.

## Critical extraction rules
- A field value is ONLY the text immediately to the right of its label (same y) OR the single line directly below it (next y, similar x).
- NEVER concatenate multiple lines into a single field value.
- NEVER include another field's label or value inside a field value.
- If a value seems too long (more than 5 words for names, more than 12 chars for dates/numbers), it is likely wrong — re-check the coordinates.
- Extract values exactly as they appear — do not interpret, translate, or reformat.
- A field value must come from a SINGLE line in the layout, not multiple lines combined.

## Document layout
Each line has normalized spatial coordinates [y=row x=column] (0.0 to 1.0).
Lines with similar y values are on the same row.
A label and its value are typically on the same row (same y, different x) or the value is on the next row directly below the label (slightly higher y, similar x).

{spatial_layout}
{mrz_section}
{hints_section}

## Step 1 — Identify document type
Determine the document type from the layout.
Available document types and their exact field names:
{schemas_json}

If the document type is not listed, use "default".

## Step 2 — Extract ONLY these missing fields
{missing_str}

For each missing field:
1. Find the label text in the layout using the label hints above
2. The value is on the SAME ROW (same y ±0.02) to the right, OR the NEXT ROW (y +0.02 to +0.06) at a similar x position
3. Take ONLY that single line as the value — never combine multiple lines
4. If the label is not found, set the value to null
5. Use exact field names from the schema only

## Step 3 — Detect inconsistencies
An inconsistency is ONLY:
- date_of_issue is after date_of_expiry
- A visual field value directly contradicts the same MRZ field
- MRZ checksum failed (flagged above if applicable)

Not inconsistencies:
- Future dates (normal for date_of_expiry)
- Missing fields
- Anything not directly verifiable from the document

## Step 4 — Verdict
genuine   → all fields consistent, MRZ valid or absent
suspicious → at least one confirmed inconsistency or MRZ checksum failed

Return ONLY raw JSON, no explanation, no markdown, no preamble:
{{
    "document_type": "<type matching schema key>",
    "fields": {{
        "<exact_field_name>": "<single line value as found or null>"
    }},
    "inconsistencies": [
        {{
            "field"      : "<field name>",
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


def analyze(output: PipelineOutput,
            config: Config,
            backend: LLMBackend) -> dict:
    fields_map = _load_fields(config.document_fields_path)

    mrz        = output.mrz
    mrz_fields = _mrz_to_dict(mrz) if mrz else {}
    mrz_valid  = mrz.valid if mrz else None

    # support both old format (list) and new format (dict with fields/label_hints)
    def get_fields(entry):
        if isinstance(entry, dict):
            return entry.get("fields", [])
        return entry

    def get_hints(entry):
        if isinstance(entry, dict):
            return entry.get("label_hints", {})
        return {}

    all_fields     = get_fields(fields_map.get("default", []))
    missing_fields = _get_missing_fields(all_fields, mrz_fields)
    label_hints    = get_hints(fields_map.get("default", {}))

    prompt = _build_prompt(
        output, fields_map, missing_fields,
        mrz_fields, mrz_valid, label_hints
    )
    raw = backend.complete(prompt)

    try:
        agent_result = _parse_response(raw)
    except Exception:
        agent_result = {"error": "agent response could not be parsed", "raw": raw}

    doc_type    = agent_result.get("document_type", "default")
    doc_entry   = fields_map.get(doc_type, fields_map.get("default", []))
    all_fields  = get_fields(doc_entry)
    label_hints = get_hints(doc_entry)

    missing_fields = _get_missing_fields(all_fields, mrz_fields)
    agent_fields   = normalize_fields(agent_result.get("fields", {}))
    merged_fields  = {**agent_fields, **mrz_fields}

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
            "fields"         : merged_fields,
            "inconsistencies": agent_result.get("inconsistencies", []),
            "verdict"        : agent_result.get("verdict"),
            "confidence"     : agent_result.get("confidence"),
            "mrz_valid"      : mrz_valid,
            "source"         : output.source,
            "confidence_avg" : output.confidence_avg
        }
    }
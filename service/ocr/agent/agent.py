import json
import re
from pathlib import Path

from service.ocr.agent.base import LLMBackend
from service.ocr.models import Config, MRZResult, PipelineOutput, TextLine
from service.ocr.normalizer import normalize_fields


def _load_template(templates_dir: str, document_type: str) -> dict | None:
    path = Path(templates_dir) / f"{document_type}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_fields_guide(template: dict) -> str:
    fields = template.get("fields", [])
    lines = []
    for f in fields:
        key       = f.get("key", "")
        label     = f.get("label", "")
        type_hint = f.get("type", "")
        lines.append(f"- {key}: look for label '{label}' (value type: {type_hint})")
    return "\n".join(lines)


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


def _build_prompt(output: PipelineOutput,
                  mrz_fields: dict,
                  mrz_valid: bool | None,
                  template: dict) -> str:
    spatial_layout = _build_spatial_layout(output.lines)

    mrz_section = ""
    if mrz_fields:
        mrz_status  = "VERIFIED (checksum passed)" if mrz_valid else "UNVERIFIED (checksum failed — possible tampering)"
        mrz_section = f"""
## MRZ reference — {mrz_status}
These values are provided ONLY for inconsistency detection in Step 3.
Do NOT copy them as extracted values. Extract every field independently from the layout.
{json.dumps(mrz_fields, ensure_ascii=False, indent=2)}
"""

    fields_guide = _build_fields_guide(template)
    step2 = f"""## Step 2 — Extract fields using the template guide below
For each field listed in the guide, find its value in the layout following the spatial rules above.
- The KEY is EXACTLY the key listed in the guide — do not invent, rename, translate, or merge keys.
- The VALUE is the verbatim text of the value line, or null if not found.
- Emit one entry per guide field; use null when the value is not present in the layout.
- Do NOT add fields that are not in the guide.

### Fields to extract
{fields_guide}"""

    return f"""You are a strict document validation agent for official identity documents.

## Document layout
Each line has normalized spatial coordinates [y=row x=column] (0.0 to 1.0). Lines with similar y (≤ 0.02 apart) are on the same row.
- A LABEL is descriptive text that names a field. A VALUE is concrete data (name, date, number, code).
- The value of a label is the FIRST line of data immediately adjacent to it, in ONE of these positions:
  1. Same row, to the IMMEDIATE RIGHT of the label (the nearest higher-x line, same y ±0.025).
  2. The IMMEDIATE NEXT row below (the row with the smallest y strictly greater than the label's y, within +0.015 to +0.05, similar x ±0.15).
- **Stop at the first immediate line.** Do NOT keep walking down rows. If the immediate next row does not contain the value, the value is null.
- Take ONE single line as the value. Extract VERBATIM (preserve casing, accents, punctuation, non-ASCII). Never combine multiple lines.
- **DO NOT replace spaces with underscores in values.** A value like "John Doe" stays as `"John Doe"`, never `"John_Doe"`. Underscores are only used in KEYS, never in VALUES.
- **The value is ONLY the data — it must NOT include the label text.** If a row contains both the label and the value (e.g. the row is "Passport No. G12345678"), the value is `"G12345678"`, NOT `"Passport No. G12345678"`. If a row reads "Date of Expiry: 08/03/2027", the value is `"08/03/2027"`, NOT the whole string.
- **Each line is consumed once.** If a line is the value of label A, it cannot also be the value of label B.
- **A line that is the translation of a label is NOT a value.** Lines that start with "/" (e.g. "/Surname and Given Name") or that consist of purely descriptive words translating the previous label belong to the LABEL group, not the value. Skip them as candidate values.
- **Be skeptical of OCR fragments.** If the candidate value looks like OCR noise (very short truncated token, partial word, isolated punctuation, or anything that doesn't read as concrete data), prefer `null` over guessing. Do not extract values like "VSA" if it's clearly a fragment of "VISA".
- If no concrete data is in the immediate adjacent position, the value is null. Never substitute with another label or a far-away line.

{spatial_layout}
{mrz_section}
## Step 1 — Document type
Infer the document type from the layout content. Use a short snake_case identifier. If you cannot determine it, use "unknown".

{step2}

## Step 3 — Detect inconsistencies
An inconsistency is ONLY:
- date_of_issue is after date_of_expiry
- A visually extracted field directly contradicts the same MRZ field
- MRZ checksum failed

Not inconsistencies:
- Future dates (normal for date_of_expiry)
- Missing or null fields
- Anything not directly verifiable from the layout

## Step 4 — Verdict
genuine    → all fields consistent, MRZ valid or absent
suspicious → at least one confirmed inconsistency or MRZ checksum failed

Return ONLY raw JSON, no explanation, no markdown, no preamble:
{{
    "document_type": "<snake_case type>",
    "fields": {{
        "<snake_case_key>": "<verbatim value or null>"
    }},
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
            backend: LLMBackend,
            template: dict) -> dict:

    mrz        = output.mrz
    mrz_fields = _mrz_to_dict(mrz) if mrz else {}
    mrz_valid  = mrz.valid if mrz else None

    prompt = _build_prompt(output, mrz_fields, mrz_valid, template)

    raw = backend.complete(prompt)

    try:
        agent_result = _parse_response(raw)
    except Exception:
        agent_result = {"error": "agent response could not be parsed", "raw": raw}

    doc_type        = agent_result.get("document_type", "unknown")
    raw_fields      = agent_result.get("fields", {}) or {}
    agent_fields    = normalize_fields({
        k: v for k, v in raw_fields.items()
        if isinstance(k, str) and not k.lower().startswith("mrz_")
    })
    inconsistencies = list(agent_result.get("inconsistencies", []) or [])

    flags: list[str] = []
    if output.mrz_unverified:
        flags.append("mrz_checksum_failed")
        inconsistencies.append({
            "field"      : "mrz",
            "description": "MRZ checksum failed — possible tampering",
        })
    if inconsistencies and "mrz_mismatch" not in flags:
        if any(
            (str(inc.get("field", "")).lower() == "mrz"
             or "mrz" in str(inc.get("description", "")).lower())
            for inc in inconsistencies
        ):
            if "mrz_checksum_failed" not in flags:
                flags.append("mrz_mismatch")

    return {
        "agent_fields": {
            "source"         : backend.__class__.__name__,
            "document_type"  : doc_type,
            "fields"         : agent_fields,
            "inconsistencies": inconsistencies,
            "confidence"     : agent_result.get("confidence"),
            "verdict"        : agent_result.get("verdict"),
            "notes"          : agent_result.get("notes")
        },

        "result": {
            "document_type"  : doc_type,
            "fields"         : agent_fields,
            "inconsistencies": inconsistencies,
            "flags"          : flags,
            "verdict"        : agent_result.get("verdict"),
            "confidence"     : agent_result.get("confidence"),
            "source"         : output.source,
            "confidence_avg" : output.confidence_avg
        }
    }
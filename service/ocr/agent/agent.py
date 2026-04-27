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


def _build_prompt(output: PipelineOutput,
                  fields_map: dict,
                  missing_fields: list,
                  mrz_fields: dict,
                  mrz_valid: bool | None) -> str:
    schemas_json   = json.dumps(fields_map, ensure_ascii=False, indent=2)
    missing_str    = json.dumps(missing_fields, ensure_ascii=False)
    spatial_layout = _build_spatial_layout(output.english_lines)

    mrz_section = ""
    if mrz_fields:
        mrz_status  = "VERIFIED (checksum passed)" if mrz_valid else "UNVERIFIED (checksum failed — possible tampering)"
        mrz_section = f"""
## MRZ extracted fields — {mrz_status}
These fields were extracted from the Machine Readable Zone and must NOT be re-extracted.
{"Use them as ground truth for cross-validation." if mrz_valid else "Treat with caution — checksum failure may indicate document tampering."}
{json.dumps(mrz_fields, ensure_ascii=False, indent=2)}
"""

    return f"""You are a strict document validation agent specialized in official identity documents.

## Document layout
Each line has normalized spatial coordinates [y=row x=column] (0.0 to 1.0).
Use coordinates to determine which value belongs to which label.
A value belongs to the label closest to it — same row (same y) or directly below (slightly higher y, similar x).
Never infer field associations from text content alone — always use spatial proximity.

{spatial_layout}
{mrz_section}
## Step 1 — Identify document type
Determine the document type from the layout.
Available document types and their exact field names:
{schemas_json}

If the document type is not listed, use "default".

## Step 2 — Extract ONLY these missing fields
{missing_str}

For each missing field:
- Find the label by text and coordinates
- Value is the text on the same row or directly below (next y, similar x)
- Extract values exactly as they appear — do not interpret or reformat
- If label not found, set to null
- Use exact field names from schema only

## Step 3 — Detect inconsistencies
An inconsistency is ONLY:
- A field value directly contradicts another field value
  (e.g. date_of_issue is after date_of_expiry)
- A visual field value contradicts the same MRZ field
- MRZ checksum failed (already flagged above if applicable)

Not inconsistencies:
- Future dates (normal for date_of_expiry)
- Missing fields
- Assumptions not verifiable from the document

## Step 4 — Verdict
genuine   → all fields consistent, MRZ valid or absent
suspicious → at least one real inconsistency or MRZ checksum failed

Return ONLY raw JSON, no explanation, no markdown:
{{
    "document_type": "<type matching schema key>",
    "fields": {{
        "<exact_field_name>": "<value as found or null>"
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

    # use mrz property — returns verified first, then unverified
    mrz        = output.mrz
    mrz_fields = _mrz_to_dict(mrz) if mrz else {}
    mrz_valid  = mrz.valid if mrz else None

    all_fields     = fields_map.get("default", [])
    missing_fields = _get_missing_fields(all_fields, mrz_fields)

    prompt = _build_prompt(output, fields_map, missing_fields, mrz_fields, mrz_valid)
    raw    = backend.complete(prompt)

    try:
        agent_result = _parse_response(raw)
    except Exception:
        agent_result = {"error": "agent response could not be parsed", "raw": raw}

    doc_type       = agent_result.get("document_type", "default")
    all_fields     = fields_map.get(doc_type, fields_map.get("default", []))
    missing_fields = _get_missing_fields(all_fields, mrz_fields)
    agent_fields   = normalize_fields(agent_result.get("fields", {}))

    # MRZ takes priority over agent in merge
    merged_fields = {**agent_fields, **mrz_fields}

    mrz_output = None
    if output.mrz_verified:
        mrz_output = {
            "valid" : True,
            "source": "mrz_verified",
            "fields": mrz_fields
        }
    elif output.mrz_unverified:
        mrz_output = {
            "valid" : False,
            "source": "mrz_unverified",
            "fields": mrz_fields,
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
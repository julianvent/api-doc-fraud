import json
import re

from rapidfuzz import fuzz

from .agent.agent import _build_fields_guide
from .backends import VisionBackend
from .models import PipelineOutput
from .normalizer import normalize_fields


_MRZ_MATCH_THRESHOLD = 90

_VLM_EXTRACT_PROMPT = """You are extracting structured data from an identity document image.

Look at the document and identify every visible label-value pair.
- A LABEL is descriptive text that names a field.
- A VALUE is concrete data (name, date, number, code) — never descriptive text.
- For each label, take only the data immediately adjacent (right or directly below).

## KEY formatting (apply ONLY to keys, never to values)
- Each KEY must be a snake_case identifier in **English**, derived from the meaning of the label.
- If the label is bilingual (e.g. "Tipo / Type", "Apellido / Surname"), use only the English portion for the key.
- If the label is in a non-English language and has no English translation on the document, translate the label's MEANING to English and use that as the key.
  Examples:
    "Apellido"                  → "surname"
    "Fecha de Nacimiento"       → "date_of_birth"
    "Lugar de Nacimiento"       → "place_of_birth"
    "Sexo"                      → "sex"
    "Nacionalidad"              → "nationality"
    "Número de Documento"       → "document_number"
    "उपमा और नाम"               → "surname_and_given_name"
- Replace every space between words with an underscore. Strip parenthetical hints and punctuation. ASCII letters, digits and underscores only.

## VALUE formatting (NEVER reformat, ONLY copy verbatim)
- Extract VALUES exactly as written on the document.
- **DO NOT replace spaces with underscores in values.** Spaces in a name like "John Doe" stay as "John Doe", never "John_Doe".
- Preserve original casing, accents, punctuation, slashes, and non-ASCII characters.
- Example: label "Date of Birth" with document text "08 March 1995"
  → key: "date_of_birth" (underscores OK in key)
  → value: "08 March 1995" (spaces preserved, NO underscores)

## Prominent standalone data (conservative capture)
Some documents have prominent data without an explicit label (e.g. a visa number "VJ9188237" printed at the top corner of a visa, an ID at the top of a passport). You MAY emit such data when ALL of these are true:
- It is visually structured (alphanumeric code, ID format, formatted number).
- It is visually prominent (large/isolated text, not surrounded by other content).
- There is no explicit label adjacent to it.

Use an inferred snake_case English key based on the document type and data pattern:
- A code prominent on a visa → "visa_number"
- A code prominent on a passport → "document_number" (only if not already captured by a labeled field)

Ignore decorative or boilerplate text: titles, country/agency names, signatures, watermarks, disclaimers, warnings, instructions. Ignore the MRZ block (long strings with `<` separators at the bottom) — it is handled separately.

## Other rules
- If a label has no visible value, set its value to null.
- Do NOT invent fields. Do NOT prefix any key with "mrz_".
- Do NOT emit duplicate keys for the same concept in different languages.

Return ONLY raw JSON, no explanation, no markdown, no preamble:
{
    "document_type": "<snake_case type>",
    "fields": {
        "<snake_case_key>": "<verbatim value or null>"
    },
    "confidence": "<high | medium | low>",
    "notes": "<critical observation only, or null>"
}"""


def _parse_response(raw: str) -> dict:
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _mrz_to_dict(mrz) -> dict:
    return normalize_fields({
        "surname"        : mrz.surname,
        "given_names"    : mrz.given_names,
        "country"        : mrz.country,
        "date_of_birth"  : mrz.birth_date,
        "date_of_expiry" : mrz.expiry_date,
        "document_number": mrz.number,
        "sex"            : mrz.sex,
    })


def _norm(value) -> str:
    return str(value).strip().upper().replace(" ", "") if value else ""


def _compare_fields_vs_mrz(fields: dict, output: PipelineOutput) -> tuple[list[dict], list[str]]:
    inconsistencies: list[dict] = []
    flags          : list[str]  = []

    mrz = output.mrz
    if mrz is None:
        return inconsistencies, flags

    if output.mrz_unverified:
        flags.append("mrz_checksum_failed")
        inconsistencies.append({
            "field"      : "mrz",
            "description": "MRZ checksum failed — possible tampering",
        })

    mrz_dict = _mrz_to_dict(mrz)
    mismatched = False
    for key, mrz_value in mrz_dict.items():
        field_value = fields.get(key)
        if not field_value or not mrz_value:
            continue
        if fuzz.ratio(_norm(field_value), _norm(mrz_value)) < _MRZ_MATCH_THRESHOLD:
            inconsistencies.append({
                "field"      : key,
                "description": f"value '{field_value}' contradicts MRZ '{mrz_value}'",
            })
            mismatched = True

    if mismatched and "mrz_mismatch" not in flags:
        flags.append("mrz_mismatch")

    return inconsistencies, flags


def extract_with_vision(image_path: str,
                        backend: VisionBackend,
                        output: PipelineOutput,
                        spatial_layout: str = "",
                        template: dict | None = None) -> dict:
    prompt = _VLM_EXTRACT_PROMPT

    if template is not None:
        fields_guide = _build_fields_guide(template)
        prompt = f"""{prompt}

## Required fields (use EXACTLY these keys, do not invent or rename)
For each field listed below, extract its value from the document image. Use the EXACT key shown — do not translate, rename, or merge keys. Use the value type hint to validate what kind of data to expect. If a field cannot be found in the image, set its value to null.

{fields_guide}

Do NOT add fields that are not listed above when emitting `fields`."""

    if spatial_layout:
        prompt = f"""{prompt}

## OCR text reference (cross-check, do not blindly copy)
The OCR engine extracted this layout from the same image. Use the image as the primary source; treat this layout only as a hint about where text appears.
{spatial_layout}"""

    raw = ""
    try:
        raw = backend.describe(image_path, prompt, max_tokens=2048)
        print(f"\n=== VLM RAW RESPONSE ({len(raw)} chars) ===\n{raw}\n=== END ===\n")
        parsed = _parse_response(raw)
        print(f"=== PARSED OK — fields keys: {list((parsed.get('fields') or {}).keys())} ===\n")
    except Exception as e:
        print(f"=== PARSE ERROR: {type(e).__name__}: {e} ===")
        print(f"=== RAW WAS:\n{raw}\n=== END ===")
        parsed = {"error": "vision response could not be parsed", "raw": raw}

    doc_type   = parsed.get("document_type") or output.document_type or "unknown"
    raw_fields = parsed.get("fields", {}) or {}
    agent_fields = normalize_fields({
        k: v for k, v in raw_fields.items()
        if isinstance(k, str) and not k.lower().startswith("mrz_")
    })

    inconsistencies, flags = _compare_fields_vs_mrz(agent_fields, output)

    verdict = "suspicious" if inconsistencies else "genuine"

    return {
        "agent_fields": {
            "source"         : backend.__class__.__name__,
            "document_type"  : doc_type,
            "fields"         : agent_fields,
            "inconsistencies": inconsistencies,
            "confidence"     : parsed.get("confidence"),
            "verdict"        : verdict,
            "notes"          : parsed.get("notes")
        },

        "result": {
            "document_type"  : doc_type,
            "fields"         : agent_fields,
            "inconsistencies": inconsistencies,
            "flags"          : flags,
            "verdict"        : verdict,
            "confidence"     : parsed.get("confidence"),
            "source"         : output.source,
            "confidence_avg" : output.confidence_avg
        }
    }

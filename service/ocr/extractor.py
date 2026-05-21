import json
import re

from .backends import VisionBackend
from .models import PipelineOutput
from .normalizer import normalize_fields
from .templates import build_fields_guide, iter_template_fields


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

## Multi-word names (CRITICAL — applies to surname, given_names, apellidos, nombres, etc.)
Many naming conventions use multiple words or even multiple lines for names. You MUST capture the FULL name, not just the first word or first line.

### Surname / Apellidos — TWO surnames is the NORM in many cultures
- **Hispanic / Portuguese / Brazilian**: people have TWO surnames (paternal + maternal). It is NORMAL and EXPECTED to see two surnames under one label.
- **Even if the label is singular ("Surname", "Apellido", "Nom")**, the VALUE on the document very often contains TWO surnames.
- Example layout 1 — both surnames on the same line:
  ```
  Surname: GARCIA LOPEZ
  ```
  → value = "GARCIA LOPEZ" (BOTH words, not just "GARCIA")
- Example layout 2 — both surnames on adjacent lines under the same label:
  ```
  Apellidos
  GARCIA
  LOPEZ
  ```
  → value = "GARCIA LOPEZ" (JOIN both lines with a single space, as a SINGLE value)
- Example layout 3 — two-column form with both surnames stacked under one column header:
  ```
  Apellidos       Nombres
  GARCIA          JUAN
  LOPEZ           CARLOS
  ```
  → surname = "GARCIA LOPEZ", given_names = "JUAN CARLOS"

**For surname / apellidos specifically: if you see TWO words or lines that look like surnames under the same label, ALWAYS include both as one value.** Never return just one.

### Given names / Nombres — also often multiple
- Compound given names are common: "JUAN CARLOS", "MARIA DEL CARMEN", "ANA SOFIA".
- Multiple given names may span lines too — join them with single spaces into one value.

### Universal rules for name fields
- Capture EVERY word that belongs to the name, joined by single spaces in their original order.
- DO NOT truncate to one word.
- DO NOT split a multi-word name across multiple keys.
- For names, joining multiple lines INTO one value is the CORRECT behavior (overrides the "never combine lines" rule which applies to non-name fields like document_number, dates, etc.).

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
        "birth_date"  : mrz.birth_date,
        "expiry_date" : mrz.expiry_date,
        "document_number": mrz.number,
        "sex"            : mrz.sex,
    })


def _norm(value) -> str:
    return str(value).strip().upper().replace(" ", "") if value else ""


_CONFIDENCE_SCORE = {"high": 0.9, "medium": 0.6, "low": 0.3}


def _confidence_to_score(confidence) -> float:
    if not confidence:
        return 0.0
    return _CONFIDENCE_SCORE.get(str(confidence).strip().lower(), 0.0)


def _compare_fields_vs_mrz(fields: dict, output: PipelineOutput) -> tuple[list[dict], list[str]]:
    inconsistencies: list[dict] = []
    flags          : list[str]  = []

    mrz = output.mrz
    print(f"[MRZ-CMP] mrz present: {mrz is not None}, mrz_unverified: {output.mrz_unverified is not None}, mrz_verified: {output.mrz_verified is not None}")
    if mrz is None:
        print(f"[MRZ-CMP] no MRZ → skipping comparison")
        return inconsistencies, flags

    if output.mrz_unverified:
        flags.append("mrz_checksum_failed")
        inconsistencies.append({
            "field"      : "mrz",
            "description": "MRZ checksum failed — possible tampering",
        })
        print(f"[MRZ-CMP] MRZ checksum FAILED → added mrz_checksum_failed flag")

    mrz_dict = _mrz_to_dict(mrz)
    print(f"[MRZ-CMP] mrz_dict keys: {list(mrz_dict.keys())}")
    print(f"[MRZ-CMP] mrz_dict values: {mrz_dict}")
    print(f"[MRZ-CMP] fields keys: {list(fields.keys())}")

    for key, mrz_value in mrz_dict.items():
        field_value = fields.get(key)
        print(f"[MRZ-CMP] check '{key}': field={field_value!r}, mrz={mrz_value!r}")
        if not field_value or not mrz_value:
            print(f"[MRZ-CMP]   → skipped (empty)")
            continue
        norm_f = _norm(field_value)
        norm_m = _norm(mrz_value)
        print(f"[MRZ-CMP]   → normalized: '{norm_f}' vs '{norm_m}'")
        if norm_f != norm_m:
            inconsistencies.append({
                "field"      : key,
                "description": f"value '{field_value}' contradicts MRZ '{mrz_value}'",
            })
            flag = f"{key} mrz_mismatch"
            if flag not in flags:
                flags.append(flag)
            print(f"[MRZ-CMP]   → MISMATCH → flag '{flag}'")
        else:
            print(f"[MRZ-CMP]   → match")

    print(f"[MRZ-CMP] final flags: {flags}")
    return inconsistencies, flags


def extract_with_vision(image_path: str,
                        backend: VisionBackend,
                        output: PipelineOutput,
                        spatial_layout: str = "",
                        template: dict | None = None) -> dict:
    prompt = _VLM_EXTRACT_PROMPT

    if template is not None:
        fields_guide = build_fields_guide(template)
        prompt = f"""{prompt}

## Required fields (PASS 1 — use EXACTLY these keys)
For each field listed below, extract its value from the document image. Use the EXACT key shown — do not rename, translate, or merge. The label on the document may be in any language (Spanish, English, Hindi, etc.); the required key already specifies the concept. Use the value type hint to validate. If after thoroughly scanning the whole image you cannot find a field, set its value to null.

{fields_guide}

## Additional fields (PASS 2 — be EQUALLY exhaustive here)
After completing Pass 1, scan the document AGAIN looking for EVERY OTHER labeled piece of data you can identify. This pass is just as important as Pass 1 — do not skip it, do not be conservative.

Include in `fields` every additional label-value pair you find that is NOT already covered by the required list above. Use English snake_case keys for these (e.g. `address`, `signature_date`, `issuing_authority`, `place_of_issue`, `father_name`, `mother_name`, `nationality_code`, `document_class`, `observations`, `endorsements`).

Be exhaustive: corners, headers, footers, side columns, multi-line addresses, anything with a clear label and a real value. Only skip purely decorative text (titles, watermarks, country names already known, signatures without an associated label, MRZ block at the bottom).

Do not invent fields with no clear label, but do NOT skip a field just because it seems minor or uncommon. The goal is a complete map of every label-value pair on the document."""

    if spatial_layout:
        prompt = f"""{prompt}

## OCR text reference (cross-check, do not blindly copy)
The OCR engine extracted this layout from the same image. Use the image as the primary source; treat this layout only as a hint about where text appears.
{spatial_layout}"""

    print(f"[OCR] calling VLM ({backend.__class__.__name__}) for extraction")
    raw = ""
    try:
        raw = backend.describe(image_path, prompt, max_tokens=2048)
        print(f"[OCR] VLM responded ({len(raw)} chars)")
        parsed = _parse_response(raw)
        print(f"[OCR] parsed OK — VLM returned keys: {list((parsed.get('fields') or {}).keys())}")
    except Exception as e:
        print(f"[OCR] VLM parse FAILED: {type(e).__name__}: {e}")
        print(f"[OCR] raw response was:\n{raw}")
        parsed = {"error": "vision response could not be parsed", "raw": raw}

    doc_type   = parsed.get("document_type") or output.document_type or "unknown"
    raw_fields = parsed.get("fields", {}) or {}
    agent_fields = normalize_fields({
        k: v for k, v in raw_fields.items()
        if isinstance(k, str) and not k.lower().startswith("mrz_")
    })

    flags: list[str] = []
    match_score = None
    extras: dict = {}

    if template is not None:
        template_keys = [f.get("key") for f in iter_template_fields(template) if f.get("key")]
        extras        = {k: v for k, v in agent_fields.items() if k not in template_keys and v}
        agent_fields  = {k: v for k, v in agent_fields.items() if k in template_keys}
        missing       = [k for k in template_keys if not agent_fields.get(k)]
        for k in missing:
            flags.append(f"{k} missing")
            agent_fields.pop(k, None)
        found_count = len(template_keys) - len(missing)
        match_score = round(found_count / len(template_keys), 3) if template_keys else None
        print(f"[OCR] template match_score: {found_count}/{len(template_keys)} = {match_score}")
        if missing:
            print(f"[OCR] missing template fields → flags: {missing}")
        if extras:
            print(f"[OCR] extras (non-template fields): {list(extras.keys())}")

    inconsistencies, mrz_flags = _compare_fields_vs_mrz(agent_fields, output)
    flags.extend(mrz_flags)
    if mrz_flags:
        print(f"[OCR] MRZ flags: {mrz_flags}")
    if inconsistencies:
        print(f"[OCR] inconsistencies: {len(inconsistencies)}")

    verdict = "suspicious" if inconsistencies else "genuine"
    confidence_score = _confidence_to_score(parsed.get("confidence"))

    return {
        "agent_fields": {
            "source"         : backend.__class__.__name__,
            "document_type"  : doc_type,
            "fields"         : agent_fields,
            "extras"         : extras,
            "inconsistencies": inconsistencies,
            "confidence"     : parsed.get("confidence"),
            "verdict"        : verdict,
            "notes"          : parsed.get("notes")
        },

        "document_type"  : doc_type,
        "fields"         : agent_fields,
        "extras"         : extras,
        "match_score"    : match_score,
        "flags"          : flags,
        "verdict"        : verdict,
        "confidence"     : parsed.get("confidence"),

        "result": {
            "document_type"  : doc_type,
            "fields"         : agent_fields,
            "extras"         : extras,
            "inconsistencies": inconsistencies,
            "flags"          : flags,
            "match_score"    : match_score,
            "verdict"        : verdict,
            "confidence"     : parsed.get("confidence"),
            "source"         : output.source,
            "confidence_avg" : confidence_score
        }
    }

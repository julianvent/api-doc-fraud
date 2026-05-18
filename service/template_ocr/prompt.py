_PROMPT_VISION_BASE = """\
You are analyzing an official identity document image (passport, visa, ID card, etc.).
Your task is to identify all data fields and classify them into two categories.

──────────────────────────────────────────────
CATEGORY DEFINITIONS
──────────────────────────────────────────────

"personal" — information about the document HOLDER:
  - Full name, surname, given names
  - Date of birth, place of birth
  - Sex / gender
  - Nationality
  - Any personal identity code (national ID, tax code, biometric code, etc.)
  - Reference to another personal document (e.g. a passport number printed on a visa
    belongs here — it identifies the person, not the current document)

"document" — information about THIS document itself:
  - Document number / identifier
  - Date of issue and date of expiry
  - Document type code
  - Issuing country or authority code
  - Number of permitted entries (for travel documents)
  - Special endorsements, remarks, or annotations
  - Visa category or type

──────────────────────────────────────────────
FIELD FORMAT
──────────────────────────────────────────────

For each field return:
  key   : snake_case identifier in English (ASCII only, lowercase)
  label : label text as it appears in the document, preferring the English version
          if bilingual; if only non-Latin script is present, transliterate to English
  type  : one of the allowed types below

Allowed types:
  "text"          → free text: names, places, nationalities, endorsement text
  "date"          → any date format (DD/MM/YYYY, DD MM YYYY, YYYY-MM-DD, etc.)
  "alphanumeric"  → document numbers, ID codes, visa numbers, personal identity codes
  "code"          → short standardized codes: country codes, document type codes,
                    visa category codes (e.g. MEX, P, S-6)
  "single_letter" → single character values: sex (M/F), document type letter
  "entry_count"   → number of permitted entries: SINGLE, DOUBLE, MULTIPLE

──────────────────────────────────────────────
DOCUMENT NUMBER DETECTION
──────────────────────────────────────────────

If you see a prominent alphanumeric code in the document that does not have an explicit
label but appears isolated in a header or corner area, it is most likely the document
number — include it under "document" with type "alphanumeric".
If a Machine Readable Zone (MRZ) is present, you may read it to confirm
or recover the document number, but do NOT include the MRZ lines themselves as fields.

──────────────────────────────────────────────
ALWAYS IGNORE — never include as fields
──────────────────────────────────────────────

  - Issuing country or government name headers
    (e.g. "ESTADOS UNIDOS MEXICANOS", "REPUBLIC OF INDIA", "GOVERNMENT OF...")
  - Document type watermarks printed as background ("VISA", "PASSPORT", "ID")
  - Operational instructions printed on the document
    (e.g. "REGISTRATION WITHIN 14 DAYS", "NOT VALID FOR PROHIBITED AREAS",
    "CHANGE OF PURPOSE NOT ALLOWED", "VALID ONLY FOR...", any restriction notice)
  - Signature line labels ("Holder's Signature", "Firma del Titular")
  - Authority or issuing officer names and office names
  - Standalone serial numbers that are clearly internal printing codes, not document IDs

{document_specific}

──────────────────────────────────────────────
OUTPUT FORMAT
──────────────────────────────────────────────

Return ONLY a valid JSON object, no explanation, no markdown:
{{
  "personal": [
    {{"key": "surname", "label": "Surname", "type": "text"}},
    {{"key": "date_of_birth", "label": "Date of birth", "type": "date"}}
  ],
  "document": [
    {{"key": "document_no", "label": "Document No.", "type": "alphanumeric"}},
    {{"key": "date_of_expiry", "label": "Date of expiry", "type": "date"}}
  ]
}}
"""

_DOC_SPECIFIC_PASSPORT = """\
This is a PASSPORT.
Bilingual labels (two languages on consecutive lines) count as ONE field — use the
English version as the label.
"""

_DOC_SPECIFIC_VISA = """\
This is a VISA.
The passport number of the holder printed on this visa is PERSONAL information
(it identifies the person, not this visa).
The visa number itself (usually top-right, alphanumeric) is DOCUMENT information.
Bilingual labels (two scripts/languages) count as ONE field — use the English version.
"""

_DOC_SPECIFIC_GENERIC = """\
Identify every label+value pair where the label describes a data field
and the value contains information specific to the document holder or the document itself.
Bilingual labels (same field in two languages) count as ONE field — prefer English.
"""

# Con Specific Prompts
#def _build_prompt(document_type: str) -> str:
#    specific = {
#        "passport": _DOC_SPECIFIC_PASSPORT,
#        "visa":     _DOC_SPECIFIC_VISA,
#    }.get(document_type.lower(), _DOC_SPECIFIC_GENERIC)
#
#    return _PROMPT_VISION_BASE.format(document_specific=specific)

# Con Generic Prompt
def _build_prompt(document_type: str) -> str:
    return _PROMPT_VISION_BASE.format(document_specific=_DOC_SPECIFIC_GENERIC)

_PROMPT_VISION_BASE = """\
You are analyzing an official identity document image (passport, visa, ID card, etc.).
Your task is to identify all data fields and classify them into two categories.

──────────────────────────────────────────────
CRITICAL RULE — NO HALLUCINATION
──────────────────────────────────────────────

Only report fields that are EXPLICITLY VISIBLE as printed text in the document image.
Do NOT infer, assume or add fields based on what you know about document types.
If a field is not visibly printed in the image, do not include it.

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
  key   : snake_case derived ONLY from the English label text — strip punctuation,
          replace spaces with underscores, lowercase. No extra normalization.
          Example: "Expiry date" → "expiry_date", "Passport No." → "passport_no"
          Exception: if the document number has no label, use the code itself as key.
  label : if the field label is bilingual, use ONLY the English portion exactly as
          it appears in the document — do not include the other language.
          If the label is only in a non-Latin script, transliterate to English.
          Do NOT translate, paraphrase or rename — copy the English text verbatim.
          Example: "Fecha de caducidad/ Expiry date" → label = "Expiry date"
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
COMPOUND LABELS
──────────────────────────────────────────────

If a single label covers multiple attributes, keep it as ONE field — do not split.
This includes labels with "and" / "y" connecting two attributes.
Example: "Surname and Given Name" → ONE field, key="surname_and_given_name", label="Surname and Given Name"
Example: "Apellidos y Nombres" → ONE field
Never return separate fields for "Surname" and "Given Name" if they share one label.

──────────────────────────────────────────────
DOCUMENT NUMBER DETECTION
──────────────────────────────────────────────

If the document has an explicit labeled field for the document number, use that field.
Only apply this fallback when NO explicit document number field exists:
  → if you see a prominent standalone alphanumeric code in a header or corner area
    with no label, it is most likely the document number — include it under "document"
    with type "alphanumeric" and use the code itself as the label.
If a Machine Readable Zone (MRZ) is present, you may read it to confirm
or recover the document number if missing, but do NOT include MRZ lines as fields.

──────────────────────────────────────────────
ALWAYS IGNORE — never include as fields
──────────────────────────────────────────────

  - Issuing country or government name headers
    (e.g. "ESTADOS UNIDOS MEXICANOS", "REPUBLIC OF INDIA", "GOVERNMENT OF...")
  - Document type watermarks printed as background ("VISA", "PASSPORT", "ID")
  - Operational instructions printed on the document
    (e.g. "REGISTRATION WITHIN 14 DAYS", "NOT VALID FOR PROHIBITED AREAS",
    "CHANGE OF PURPOSE NOT ALLOWED", "VALID ONLY FOR...", any restriction notice)
  - Signature line labels ("Holder's Signature", "Firma del Titular", etc.)
  - Authority labels and everything associated with them: any label containing
    "Authority", "Autoridad", "Issuing Officer", "Signed by" or similar,
    AND any field whose visible value is a person's full name acting in official
    capacity or the name of a government office / department
    (e.g. "NAIA ANALEAH MENDEZ GOU", "OF. PASAPORTES YUCATAN") — ignore both
    the label and its value entirely
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
    {{"key": "passport_no", "label": "Passport No", "type": "alphanumeric"}},
    {{"key": "expiry_date", "label": "Expiry date", "type": "date"}}
  ]
}}
"""

_DOC_SPECIFIC_PASSPORT = """\
This is a PASSPORT.
Bilingual labels (two languages on consecutive lines) count as ONE field — use the
English version as the label exactly as printed.
"""

_DOC_SPECIFIC_VISA = """\
This is a VISA.
The passport number of the holder printed on this visa is PERSONAL information
(it identifies the person, not this visa).
The visa number itself (usually top-right, alphanumeric) is DOCUMENT information.
Bilingual labels (two scripts/languages) count as ONE field — use the English version as the label exactly as printed.
Date of issue and date of expiry are DOCUMENT fields
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

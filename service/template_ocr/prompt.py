_PROMPT_VISION_BASE = """\
You are analyzing an official identity document image (passport, visa, ID card, etc.).
Identify all data fields and classify them into two categories: "personal" and "document".
 
──────────────────────────────────────────────
CRITICAL — ONLY WHAT IS VISIBLE
──────────────────────────────────────────────
 
Only include fields that are EXPLICITLY VISIBLE as printed text in the image.
Do NOT infer or add fields based on document type knowledge.
 
──────────────────────────────────────────────
CATEGORIES
──────────────────────────────────────────────
 
"personal" — about the document HOLDER:
  names, birth date, birth place, sex, nationality, personal identity codes,
  references to the holder's other documents (e.g. passport number on a visa)
 
"document" — about THIS document:
  document number, issue date, expiry date, document type, issuing country code,
  number of entries, visa type/category, special endorsements or remarks
 
──────────────────────────────────────────────
KEY NORMALIZATION
──────────────────────────────────────────────

Use these exact standard keys whenever the field matches — regardless of language or wording:
 
  Personal:
    surname           → if surname / family name / apellidos appears as a separate field
    given_names       → if given names / nombres / first name appear as a separate field
    surname_and_given_names → if both appear combined in a single field label
    birth_date        → date of birth / fecha de nacimiento
    sex               → sex / sexo / gender
    nationality       → nationality / nacionalidad
    place_of_birth    → place of birth / lugar de nacimiento
    personal_id       → any personal identity code (CURP, Aadhaar, PAN, national ID, etc.)
 
  Document:
    document_number   → document No. / passport No. / visa No. / any document identifier
    date_of_issue     → date of issue / fecha de expedición / issue date
    expiry_date       → expiry date / fecha de caducidad / date of expiry / valid until
    document_type     → document type code / tipo
    issuing_country   → issuing country code / clave del país / country code
    visa_type         → visa type / visa category
    no_of_entries     → number of entries / no. of entries
    special_endorsement → special endorsement / endorsements / remarks / observaciones
 
For any field that does not match the above, derive the key from the English label:
snake_case, strip punctuation, lowercase.
 
If a label is bilingual, use ONLY the English portion as label and to derive the key.
 
──────────────────────────────────────────────
COMPOUND LABELS
──────────────────────────────────────────────
 
A label combining multiple attributes (e.g. "Surname and Given Name") is ONE field.
Do NOT split it — use key="surname_and_given_names", label="Surname and Given Name".
 
──────────────────────────────────────────────
DOCUMENT NUMBER
──────────────────────────────────────────────
 
If the document number has no explicit label but appears as a prominent standalone
alphanumeric code (header or corner area), include it as document_number.
If MRZ is present, read it to confirm or recover the document_number if not found
elsewhere — but do NOT include the MRZ lines themselves as fields.
 
──────────────────────────────────────────────
ALWAYS IGNORE
──────────────────────────────────────────────
 
  - Country / government name headers
  - Document type watermarks (background text)
  - Operational instructions and restriction notices
  - Signature labels and holder signature areas
  - Authority labels and their values (officer names, office names)
  - Internal serial / printing codes
 
{document_specific}
 
──────────────────────────────────────────────
OUTPUT
──────────────────────────────────────────────
 
In addition to the fields, produce two extra blocks:

──────────────────────────────────────────────
FINGERPRINT
──────────────────────────────────────────────

"fingerprint.layout_desc" — one or two sentences describing the document layout:
  document family (passport / visa / ID card / etc.), MRZ presence and approximate
  position, dominant color, photo position, distinctive structural elements.

"fingerprint.anchors" — 3 to 6 short verbatim strings printed on the document that
  are STABLE across instances (issuing authority name, document title, country
  header). Do NOT include personal data, dates, or document numbers.

──────────────────────────────────────────────
VALIDATORS
──────────────────────────────────────────────

"validators" — OPTIONAL dict mapping field key to declarative validation rules.
Only include rules you can infer with HIGH confidence from the document type
(do NOT invent rules). Allowed shapes:
  {{"regex": "<pattern>"}}                                   for alphanumeric formats
  {{"type": "date", "must_be_past": true}}                   for birth_date and similar
  {{"type": "date", "must_be_future": true}}                 for expiry_date
  {{"type": "date", "before": "<other_field_key>"}}          for cross-field
Examples:
  document_number → {{"regex": "^[A-Z][0-9]{{8}}$"}}
  birth_date      → {{"type": "date", "must_be_past": true}}
  expiry_date     → {{"type": "date", "must_be_future": true}}

If you cannot infer any high-confidence rule, return an empty dict for "validators".

──────────────────────────────────────────────
OUTPUT
──────────────────────────────────────────────

Return ONLY valid JSON, no explanation, no markdown:
{{
  "personal": [
    {{"key": "surname", "label": "Surname", "type": "text"}},
    {{"key": "birth_date", "label": "Date of birth", "type": "date"}}
  ],
  "document": [
    {{"key": "document_number", "label": "Passport No.", "type": "alphanumeric"}},
    {{"key": "expiry_date", "label": "Expiry date", "type": "date"}}
  ],
  "fingerprint": {{
    "layout_desc": "Passport-type document, MRZ TD3 bottom, burgundy color, photo top-left.",
    "anchors": ["ESTADOS UNIDOS MEXICANOS", "PASAPORTE", "SRE"]
  }},
  "validators": {{
    "document_number": {{"regex": "^[A-Z][0-9]{{8}}$"}},
    "expiry_date":     {{"type": "date", "must_be_future": true}}
  }}
}}

Allowed field types: "text", "date", "alphanumeric", "code", "single_letter", "entry_count"
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
Bilingual labels (same field in two languages) count as ONE field — use English portion.
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

_PROMPT_VISION_BASE = """\
You are analyzing an official document image.
Identify all data fields visible in the document.

For each field return:
- key   : snake_case identifier derived from the label (ASCII only, lowercase)
- label : the exact label text as it appears in the document (preserve original language)
- type  : the data type of the value this field holds

Allowed types:
  "text"          → free text: names, places, nationalities
  "date"          → any date format (DD/MM/YYYY, DD MM YYYY, Month DD YYYY, etc.)
  "alphanumeric"  → ID numbers, passport numbers, visa numbers, CURP codes
  "code"          → short codes: country codes (MEX), visa type codes (S-6), category codes
  "single_letter" → single letter values: document type (P), sex (M/F)
  "entry_count"   → number of entries: SINGLE, DOUBLE, MULTIPLE

{document_specific}

ALWAYS IGNORE — do not include:
  - Country or government name headers (e.g. "ESTADOS UNIDOS MEXICANOS", "REPUBLIC OF INDIA")
  - Document type watermarks ("VISA", "PASSPORT" as background text)
  - Legal disclaimers and restriction notices
  - Signature area labels, authority names, office names
  - Serial numbers printed alone without a descriptive label
  - MRZ lines at the bottom

Return ONLY a valid JSON object, no explanation, no markdown:
{{
  "fields": [
    {{"key": "passport_no", "label": "Pasaporte No.", "type": "alphanumeric"}},
    {{"key": "fecha_de_nacimiento", "label": "Fecha de nacimiento", "type": "date"}}
  ]
}}
"""

_DOC_SPECIFIC_PASSPORT = """\
This is a MEXICAN PASSPORT. Expected fields include:
  Tipo / Type (single_letter), Clave del país / Issuing state code (code),
  Pasaporte No. / Passport No. (alphanumeric), Apellidos / Surname (text),
  Nombres / Given names (text), Nacionalidad / Nationality (text),
  Fecha de nacimiento / Date of birth (date), CURP / Personal No. (alphanumeric),
  Sexo / Sex (single_letter), Lugar de nacimiento / Place of birth (text),
  Fecha de expedición / Date of issue (date), Fecha de caducidad / Expiry date (date).
Bilingual labels (Spanish + English on consecutive lines) count as ONE field.
"""

_DOC_SPECIFIC_VISA = """\
This is an INDIAN VISA. Expected fields include:
  Visa number top-right (alphanumeric),
  उपनाम और नाम / Surname and Given Name (text),
  पामपाटमज्या / Passport No (alphanumeric),
  वीजा टाईप / Visa Type (code),
  प्रवेशों की संख्या / No Of Entries (entry_count),
  जारी करने की तिथि / Date of Issue (date),
  समाप्ति की तिथि / Date of Expiry (date),
  विशेष पृष्ठांकन / Special Endorsement (text).
Bilingual labels (Hindi + English) count as ONE field.
"""

_DOC_SPECIFIC_GENERIC = """\
Identify every label+value pair where the label describes a data field
and the value contains information specific to the document holder.
Bilingual labels (same field in two languages) count as ONE field.
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

_PARSE_PROMPT = """\
Please output the layout information from the document image, \
including each layout element's bbox, its category, and the corresponding \
text content within the bbox.
1. Bbox format: [x1, y1, x2, y2]
2. Layout Categories: The possible categories are \
['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', \
'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].
3. Text Content: Extract the complete text within each bbox, \
preserving the original language and script.
4. Reading Order: Elements must be listed in natural reading order.
5. Final Output: The entire output must be a single JSON object.\
"""

_PROMPT_BASE = """\
You are analyzing a structured official document.
Below is a numbered list of text elements detected by OCR, each with its position.

OCR Elements:
{elements}

ALWAYS classify as ANCHOR — never include in fields:
  - Country or government name in the document header, even if bilingual 
    (e.g. "भारत गणराज्य REPUBLIC OF INDIA", "ESTADOS UNIDOS MEXICANOS")
    RULE: if it names the issuing country or government → always anchor
  - Document type watermarks ("VISA", "PASSPORT" as background)
  - Legal disclaimers containing words like: "not allowed", "prohibited", 
    "restricted", "within N days", "not valid for", "cantonment"
  - Signature line labels, office names, authority names
  - Page numbers or serial codes with no corresponding label

CRITICAL: When in doubt between anchor and field → classify as FIELD.

{document_specific}

Use position [top, left, bottom, right] to pair labels with values.
Value is directly below (value top ≈ label bottom, similar left)
or directly to the right (similar top, value left ≈ label right).

Return ONLY a valid JSON object, no explanation, no markdown:
{{
  "anchors": [{{"text": "...", "idx": 0}}],
  "fields": [{{"label_text": "...", "label_idx": 0, "value_text": "...", "value_idx": 1}}]
}}
If label and value are in the same element, use the same index for both.
"""

# Prompts genericos pero orientados a cada tipo de documento
_PROMPT_VISA_GENERIC = """\
ALWAYS classify as FIELD for this VISA document:
  - Bilingual field labels in format "Hindi text /English text" 
    (e.g. "पासपोर्टसंख्या /Passport No", "वीजा टाइप /Visa Type")
    These are field labels, NOT anchors, even though they are bilingual
  - The visa number in the top area (alphanumeric code like "VJ 9188237") 
    IS a field — classify the label above it and the code as its value
  - Short data values below or next to labels (names, dates, codes, numbers)

IGNORE for this VISA document (classify as anchor):
  - "भारत गणराज्य REPUBLIC OF INDIA" — country header, always anchor
  - Any text about registration requirements or purpose restrictions
  - The large "VISA" watermark text
"""

_PROMPT_PASSPORT_GENERIC = """\
ALWAYS classify as FIELD for this PASSPORT document:
  - Bilingual labels where Spanish and English appear on separate lines 
    one after the other (e.g. "Apellidos" then "Surname" on next line)
    → treat as ONE field, combine as "Apellidos / Surname"
  - The header section contains real fields: document type label, 
    issuing country code label, and passport number label — these ARE fields
  - Short codes like "P", "MEX", and alphanumeric passport numbers 
    are VALUES, not labels

IGNORE for this PASSPORT document (classify as anchor):
  - "ESTADOS UNIDOS MEXICANOS" — country header, always anchor
  - "PASAPORTE / PASSPORT" printed on the left side — document type watermark
  - "Observaciones / Remarks" section
  - "Firma del Titular / Holder's Signature" — signature label
  - "Autoridad / Authority" — authority label
  - Authority person name and office name below the signature area
  - The serial number printed alone on the bottom left (e.g. "551072")
"""

_PROMPT_GENERIC = """\
ALWAYS classify as FIELD:
  - Any label+value pair where the label identifies a data field
    and the value contains information specific to the document holder
  - Bilingual labels (same field in two languages) count as ONE field
"""


_PROMPT_PASSPORT_EXPLICIT = """\
ALWAYS classify as FIELD for this PASSPORT document:
  - Bilingual labels where Spanish and English appear on separate lines
    (e.g. "Apellidos" followed by "Surname") → treat as ONE field
  - Short codes like "P", "MEX", and alphanumeric passport numbers are VALUES, not labels

Extract EXACTLY these 12 fields — no more, no less:
  - Tipo / Type → value: single letter document type code
  - Clave del país de expedición / Issuing state code → value: 3-letter country code
  - Pasaporte No. / Passport No. → value: alphanumeric passport number
  - Apellidos / Surname → value: holder's surnames
  - Nombres / Given names → value: holder's given names
  - Nacionalidad / Nationality → value: nationality text
  - Fecha de nacimiento / Date of birth → value: birth date
  - CURP / Personal No. → value: CURP alphanumeric code
  - Sexo / Sex → value: single letter sex code
  - Lugar de nacimiento / Place of birth → value: place name
  - Fecha de expedición / Date of issue → value: issue date
  - Fecha de caducidad / Expiry date → value: expiry date

PAIRING RULES:
  - Every value is directly below or directly to the right of its label
  - Single letters ("P", "M") and short codes ("MEX") are VALUES, not labels
  - Dates ("04 02 2022", "04 02 2028") are VALUES, not labels
  - The long alphanumeric CURP code (e.g. "TUEA030527HYNRNNA2") is a VALUE, not a label
 
CRITICAL — fields that MUST have DIFFERENT value_idx from each other:
  - Field 8 (CURP) and Field 9 (Sexo) are in DIFFERENT rows → assign DIFFERENT value_idx
  - Field 10 (Lugar de nacimiento) and Field 12 (Fecha de caducidad) are side-by-side
    in the same row → assign DIFFERENT value_idx (one is on the left, one on the right)
  - Field 11 (Fecha de expedición) and Field 12 (Fecha de caducidad) are DIFFERENT dates
    in DIFFERENT rows → assign DIFFERENT value_idx
"""

_PROMPT_VISA_EXPLICIT = """\
ALWAYS classify as FIELD for this VISA document.
Extract EXACTLY these fields if present — no more, no less:
  - Visa number (alphanumeric code in top area like "VJ 9010101") → label is the code itself, value is the repeated code below
  - उपनाम और नाम /Surname and Given Name → value: full name of holder
  - पामपाटमज्या /Passport No → value: passport number
  - वीजा टाईप /Visa Type → value: visa type code (e.g. S-6)
  - प्रवेशों की संख्या No Of Entries → value: number of entries (e.g. DOUBLE)
  - जारी करने की तिथि /Date of Issue → value: issue date
  - समाप्ति की तिथि /Date of Expiracy → value: expiry date
  - विशेष पृष्ठांकन /Special Endorsement → value: endorsement text below it

PAIRING RULES — CRITICAL:
  - Dates like "11/01/2026", "10/08/2026" → always VALUES, NEVER labels
  - Codes like "S-6", "DOUBLE" → always VALUES, NEVER labels
  - Passport numbers like "N08181818" → always a VALUE for the "Passport No" label above it
  - NEVER create a field where label_text is a date, a number, or a short alphanumeric code
  - The label is always the Hindi/English descriptive text ABOVE or to the LEFT of the value
  - If you see a date or code as a standalone element, look for its label above it

IMPORTANT — some elements contain both label and value separated by \\n:
  Example: "जगे कानेकी तिथि /Date of Issue\\n11/11/2011"
  → label_text should be ONLY "जगे कानेकी तिथि /Date of Issue"
  → value_text should be ONLY "11/11/2011"
  → use the SAME index for both label_idx and value_idx
  Never include the value in the label_text field.

STRICTLY IGNORE — do NOT include under any circumstances:
  - Any text in Devanagari script that is NOT one of the labels listed above
  - "भारत गणराज्य REPUBLIC OF INDIA" or any country/government name
  - "भारत गणराज्य REPUBLIC OF INDIA" — this is the country name header, NOT a field
  - "VISA" watermark
  - Legal disclaimers or restriction notices
  - MRZ lines
"""

def _build_prompt(elements_text: str, document_type: str, explicit: bool = False,) -> str:
    if explicit:
        specific = {
            "visa":     _PROMPT_VISA_EXPLICIT,
            "passport": _PROMPT_PASSPORT_EXPLICIT,
        }.get(document_type.lower(), _PROMPT_GENERIC)
    else:
        specific = {
            "visa":     _PROMPT_VISA_GENERIC,
            "passport": _PROMPT_PASSPORT_GENERIC,
        }.get(document_type.lower(), _PROMPT_GENERIC)

    return _PROMPT_BASE.format(
        elements=elements_text,
        document_specific=specific,
    )
_PROMPT_VISION_BASE = """\
You are analyzing an official document image. The document belongs to ONE of two
families:
  - IDENTITY document: passport, visa, ID card, driver's license, residence
    permit, PAN/Aadhaar/national ID, etc. — identifies a person.
  - PROOF-OF-ADDRESS document: utility bill (electricity, water, gas, internet,
    phone), bank statement, lease/rental contract, tax notice, government
    correspondence — proves where the holder lives.

Identify all data fields present and classify them into two categories:
"personal" and "document". The two categories apply to BOTH families.
 
──────────────────────────────────────────────
CRITICAL — ONLY WHAT IS VISIBLE
──────────────────────────────────────────────

Only include fields that are EXPLICITLY VISIBLE as printed text in the image.
Do NOT infer or add fields based on document type knowledge.
Conversely, do NOT skip a visible labeled field just because it is uncommon —
capture every label-value pair you can actually see, scanning the whole document
(top, bottom, side panels). There is no minimum or maximum number of fields:
return exactly what is printed, no more, no less.

──────────────────────────────────────────────
LABEL vs VALUE — DO NOT CONFUSE THEM
──────────────────────────────────────────────

For each field, you emit a "label" (descriptive text) and a "key" (snake_case
identifier). You do NOT emit the value here — this is a TEMPLATE.

A LABEL is a noun phrase that NAMES the kind of data:
  "Account Number", "Date of Birth", "Total a pagar", "Periodo Facturado",
  "Surname", "Domicilio", "Customer Name", "Due Date", "Apellidos".

A VALUE is the concrete instance data:
  "123456789", "1990-05-12", "$1,250.00", "01-MAR al 31-MAR",
  "GARCIA", "AV. RIO NAZAS 123", "JUAN CARLOS PEREZ", "31/12/2026".

CRITICAL: never emit a value as a label. If your "label" is a number, date,
monetary amount, address, or a person's name, you are confusing them — the
real label is the descriptive text usually printed to the LEFT, ABOVE, or
BEFORE A COLON next to that value. Re-look at the image and use that text.

  ✗ WRONG: {{"key": "01_mar_al_31_mar", "label": "01-MAR al 31-MAR"}}
    (the model captured the value as the label)
  ✓ RIGHT: {{"key": "billing_period",   "label": "Periodo Facturado"}}

  ✗ WRONG: {{"key": "1250_00", "label": "$1,250.00"}}
  ✓ RIGHT: {{"key": "amount_due", "label": "Total a pagar"}}

If a piece of data appears WITHOUT any descriptive label nearby (a prominent
standalone code), see the DOCUMENT NUMBER section below — that is the only
case where you create a synthetic label like "Document number".
 
──────────────────────────────────────────────
CATEGORIES
──────────────────────────────────────────────
 
"personal" — about the HOLDER / account holder (the person the document belongs to):
  - identity docs: names, birth date, birth place, sex, nationality,
    personal identity codes, references to the holder's other documents
    (e.g. passport number on a visa)
  - proof-of-address docs: holder/customer name, service or billing address,
    contact info (phone, email) when printed as a field

"document" — about THIS specific document instance:
  - identity docs: document number, issue date, expiry date, document type,
    issuing country code, number of entries, visa type/category, endorsements
  - proof-of-address docs: account or service number, billing period or period
    covered, issue/bill date, due date, amount due / total, previous balance,
    consumption (kWh / m3 / GB), tariff, issuing authority reference codes
 
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
    address           → service address / billing address / domicilio (proof-of-address)
    phone             → telephone / contact number / teléfono (proof-of-address)
    email             → email address / correo electrónico (proof-of-address)

  Document:
    document_number   → document No. / passport No. / visa No. / any document identifier
    date_of_issue     → date of issue / fecha de expedición / issue date
    expiry_date       → expiry date / fecha de caducidad / date of expiry / valid until
    document_type     → document type code / tipo
    issuing_country   → issuing country code / clave del país / country code
    visa_type         → visa type / visa category
    no_of_entries     → number of entries / no. of entries
    special_endorsement → special endorsement / endorsements / remarks / observaciones
    account_number    → account no. / customer no. / número de cuenta / número de servicio (proof-of-address)
    billing_period    → billing period / período / period covered (proof-of-address)
    bill_date         → bill date / issue date / fecha de emisión (proof-of-address)
    due_date          → due date / payment due / fecha límite de pago (proof-of-address)
    amount_due        → total due / amount due / total a pagar (proof-of-address)
    consumption       → consumption / kWh / m3 / GB / consumo (proof-of-address)
 
For any field that does not match the above, derive the key from the English label:
snake_case, strip punctuation, lowercase.

If a label is bilingual, use ONLY the English portion as label and to derive the key.
 
──────────────────────────────────────────────
COMPOUND LABELS
──────────────────────────────────────────────
 
DEFAULT: emit SEPARATE fields. Only merge into a combined field when the
document prints a SINGLE COMBINED VALUE covering several concepts.

  ✗ WRONG — merging when the values are separate:
    The doc prints "Surname" with value "GARCIA LOPEZ" on one line,
    and "Given Names" with value "JUAN CARLOS" on the next line, under
    a shared section header "Surname and Given Names".
    → DO NOT emit a single field "surname_and_given_names".
    → Emit TWO fields: surname and given_names (use KEY NORMALIZATION).

  ✓ RIGHT — merging only when there is literally one combined value:
    The doc prints "Place and date of birth: New York, 1990-05-12"
    as a single text block with no internal separation.
    → ONE field: key="place_and_date_of_birth".

For passports specifically: surname and given_names are ALWAYS two separate
fields under ICAO conventions, even if a section header above them mentions
both. NEVER emit a single "surname_and_given_names" field for a passport.
 
──────────────────────────────────────────────
DOCUMENT NUMBER
──────────────────────────────────────────────

If the document number has no explicit label but appears as a prominent standalone
alphanumeric code (header or corner area), include it as document_number with
label="Document number" (NEVER use the value itself as the label).
If MRZ is present, read it to confirm or recover the document_number if not found
elsewhere — but do NOT include the MRZ lines themselves as fields.

──────────────────────────────────────────────
ADDRESS (PROOF-OF-ADDRESS — MANDATORY EXCEPTION)
──────────────────────────────────────────────

THIS IS AN EXPLICIT EXCEPTION TO THE "ONLY VISIBLE LABELS" RULE — read carefully.

For proof-of-address documents (utility bill, bank statement, lease, tax notice,
etc.) the holder's SERVICE / BILLING ADDRESS is the document's primary purpose
and MUST appear as a field — it is non-negotiable, the document exists to
prove it.

The address frequently appears WITHOUT a printed label, as a multi-line block
right under the customer/holder name (street + colonia/neighborhood + city +
postal code). You MUST emit it as a field even when no label is visible next
to it — exactly like the DOCUMENT NUMBER rule above is an exception for
identity docs.

Emit:
  - key:   "address"
  - label: the printed label if any ("Domicilio", "Service address",
           "Dirección", "Calle", etc.); OTHERWISE use label="Address".
  - type:  "text"

If you find yourself NOT emitting an address field on a proof-of-address
document, you are wrong — go back and find the multi-line block under the
customer name and emit it.
 
──────────────────────────────────────────────
ALWAYS IGNORE
──────────────────────────────────────────────
 
  - Country / government / company name headers (top-of-document title)
  - Document type watermarks (background text)
  - Long operational instructions / restriction notices / paragraphs of legal text
  - Signature labels and holder signature areas
  - Authority labels and their values (officer names, office names) — but a
    proof-of-address "issuing_authority" code IS a valid field
  - Internal serial / printing codes (tiny edge prints)
  - Marketing / promotional text on bills
 
{document_specific}
 
──────────────────────────────────────────────
OUTPUT
──────────────────────────────────────────────
 
In addition to the fields, produce two extra blocks:

──────────────────────────────────────────────
FINGERPRINT
──────────────────────────────────────────────

"fingerprint.layout_desc" — one or two sentences describing the document layout:
  document family (passport / visa / ID card / utility bill / bank statement /
  lease / etc.), MRZ presence and approximate position (only for identity docs),
  dominant color, photo position (if any), distinctive structural elements
  (header band, table of charges, billing summary, etc.).

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

Before returning, for EACH field in your output verify you can point to its
printed label on the image. If you cannot, REMOVE that field — do not pad the
output with fields that are not actually printed (the example below is structural
only; its keys are illustrative, not required).

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

Allowed field types: "text", "date", "alphanumeric", "code", "single_letter",
                     "entry_count", "decimal", "currency"
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

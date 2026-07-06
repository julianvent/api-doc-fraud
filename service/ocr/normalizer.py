from datetime import datetime, date

DATE_FORMATS = [
    "%y%m%d",       # MRZ: 900101
    "%Y%m%d",       # 19900101
    "%d/%m/%Y",     # 01/01/1990
    "%d/%m/%y",     # 01/01/90
    "%d-%m-%Y",     # 01-01-1990
    "%d-%m-%y",     # 01-01-90
    "%d.%m.%Y",     # 01.01.1990
    "%d.%m.%y",     # 01.01.90
    "%d %m %Y",     # 01 01 1990 (space-separated OCR artifact)
    "%d %m %y",     # 01 01 90
    "%Y/%m/%d",     # 1990/01/01
    "%Y-%m-%d",     # 1990-01-01 (ISO)
    "%Y.%m.%d",     # 1990.01.01
    "%Y %m %d",     # 1990 01 01
    "%d %b %Y",     # 01 JAN 1990
    "%d %B %Y",     # 01 January 1990
    "%d/%b/%Y",     # 01/JAN/1990
    "%d-%b-%Y",     # 01-JAN-1990
    "%b %d %Y",     # JAN 01 1990
    "%B %d %Y",     # January 01 1990
    "%b %d, %Y",    # JAN 01, 1990
    "%B %d, %Y",    # January 01, 1990
]

# Canonical English date field names. Bilingual keys (e.g. "fecha_de_nacimiento_date_of_birth")
# are matched by suffix — see _is_date_field().
DATE_FIELDS = {
    "date_of_birth",
    "date_of_issue",
    "date_of_expiry",
    "birth_date",
    "expiry_date",
    "issue_date",
}

_OUTPUT_FORMAT = "%Y/%m/%d"


def normalize_date(value) -> str | None:
    """Normalize any date representation to YYYY/MM/DD.
    Accepts str, datetime and date. Returns the original string value if
    the format is not recognized."""
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.strftime(_OUTPUT_FORMAT)
    clean = str(value).strip().upper()
    if not clean:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(clean, fmt).strftime(_OUTPUT_FORMAT)
        except ValueError:
            continue
    return str(value)


def _is_date_field(key: str) -> bool:
    """True when key is a known date field — exact match or bilingual suffix.
    Template keys like 'fecha_de_nacimiento_date_of_birth' end with a canonical
    DATE_FIELDS name preceded by an underscore."""
    return key in DATE_FIELDS or any(
        key.endswith(f"_{df}") for df in DATE_FIELDS
    )


def normalize_fields(fields: dict) -> dict:
    """Apply normalize_date to date fields; leave all other fields unchanged.
    Handles both exact keys ('date_of_birth') and bilingual template keys
    ('fecha_de_nacimiento_date_of_birth') via suffix matching."""
    result = {}
    for key, value in fields.items():
        if value is None:
            result[key] = None
        elif _is_date_field(key):
            result[key] = normalize_date(value)
        else:
            result[key] = value
    return result

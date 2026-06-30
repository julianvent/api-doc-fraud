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
    "%d %m %Y",
]

DATE_FIELDS = {
    "date_of_birth",
    "date_of_issue",
    "date_of_expiry",
    "birth_date",
    "expiry_date",
    "issue_date"
}


def normalize_date(value) -> str | None:
    """Normalize any date representation to dd/mm/yyyy.
    Accepts str, datetime and date. Returns the original value if the format is not recognized."""
    if value is None:
        return None
    # datetime/date objects: format directly to avoid str() adding a time component
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y/%m/%d")
    clean = str(value).strip().upper()
    if not clean:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(clean, fmt).strftime("%Y/%m/%d")
        except ValueError:
            continue
    return str(value)  # unrecognized format: return original


def normalize_fields(fields: dict) -> dict:
    """Apply normalize_date to every value. Non-date values are returned unchanged."""
    result = {}
    for key, value in fields.items():
        if value is None:
            result[key] = None
        else:
            result[key] = normalize_date(value)
    return result

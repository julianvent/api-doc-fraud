from datetime import datetime

DATE_FORMATS = [
    "%y%m%d",       # MRZ: 900101
    "%d/%m/%Y",     # 01/01/1990
    "%d-%m-%Y",     # 01-01-1990
    "%d %b %Y",     # 01 JAN 1990
    "%d %B %Y",     # 01 January 1990
    "%Y%m%d",       # 19900101
    "%d.%m.%Y",     # 01.01.1990
    "%b %d, %Y",    # JAN 01, 1990
    "%B %d, %Y",    # January 01, 1990
    "%Y-%m-%d",     # 1990-01-01 (already ISO)
]

DATE_FIELDS = {
    "date_of_birth",
    "date_of_issue",
    "date_of_expiry",
    "birth_date",
    "expiry_date",
    "issue_date"
}


def normalize_date(value: str) -> str | None:
    if not value:
        return None
    clean = value.strip().upper()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(clean, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return value


def normalize_fields(fields: dict) -> dict:
    result = {}
    for key, value in fields.items():
        if value is None:
            result[key] = None
        elif key in DATE_FIELDS:
            result[key] = normalize_date(str(value))
        else:
            result[key] = value
    return result
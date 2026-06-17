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
    "%d %m %Y",     # 01 01 1990  ← OCR con espacios
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

DATE_FIELDS = {
    "date_of_birth",
    "date_of_issue",
    "date_of_expiry",
    "birth_date",
    "expiry_date",
    "issue_date"
}


def normalize_date(value) -> str | None:
    """Normaliza cualquier representación de fecha a dd/mm/yyyy.
    Acepta str, datetime y date. Devuelve el valor original si no reconoce el formato."""
    if value is None:
        return None
    # objetos datetime/date — conversión directa sin pasar por str()
    if isinstance(value, (datetime, date)):
        return value.strftime("%d/%m/%Y")
    clean = str(value).strip().upper()
    if not clean:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(clean, fmt).strftime("%d/%m/%Y")
        except ValueError:
            continue
    return str(value)  # no reconocido: devolver original


def normalize_fields(fields: dict) -> dict:
    """Aplica normalize_date a todos los valores: si el valor no es una fecha
    reconocible, normalize_date devuelve el original sin modificar."""
    result = {}
    for key, value in fields.items():
        if value is None:
            result[key] = None
        else:
            result[key] = normalize_date(value)
    return result

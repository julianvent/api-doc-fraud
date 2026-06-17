import re
from .models import TextLine, MRZResult

MRZ_PATTERN = re.compile(r'^[A-Z0-9<]{30,44}$')

# Longitudes exactas por formato MRZ
_MRZ_LENGTHS   = (44, 36, 30)
_LENGTH_SNAP   = 3   # tolerancia ±chars para aceptar una línea con ruido OCR
_BOTTOM_BAND   = 0.25  # zona inferior del documento donde vive la MRZ (normalizado)

# Caracteres que OCR confunde frecuentemente con '<'
_OCR_FIXUPS = str.maketrans({
    "(": "<", "[": "<", "{": "<",
    ">": "<", "/": "<", "—": "<",
    "«": "<", "‹": "<", "＜": "<",
    "«": "<",  # «
    "‹": "<",  # ‹
})


def _normalize_mrz_text(text: str) -> str:
    """Normaliza una línea OCR para compararla con formato MRZ."""
    return (
        text.strip()
        .upper()
        .translate(_OCR_FIXUPS)
        .replace(" ", "")   # OCR a veces añade espacios dentro de la MRZ
    )


def _snap_length(text: str) -> str | None:
    """Ajusta el texto a la longitud MRZ más cercana si está dentro de tolerancia.
    Retorna la línea paddeada/truncada, o None si está demasiado lejos."""
    n    = len(text)
    best = min(_MRZ_LENGTHS, key=lambda l: abs(l - n))
    if abs(best - n) <= _LENGTH_SNAP:
        return text.ljust(best, "<")[:best]
    return None


def _is_mrz_candidate(text: str) -> bool:
    """True si el texto (ya normalizado) parece una línea MRZ."""
    return bool(re.match(r'^[A-Z0-9<]{20,50}$', text)) and "<" in text


def _line_bottom_center(line: TextLine) -> float | None:
    """Centro Y normalizado (0-1) de la línea. None si no hay bbox."""
    if line.bbox is None or len(line.bbox) == 0:
        return None
    ys = [pt[1] for pt in line.bbox]
    return (min(ys) + max(ys)) / 2


def _find_lines(lines: list[TextLine]) -> list[str]:
    # 1. Filtrar a la banda inferior del documento
    bottom = [l for l in lines if (_cy := _line_bottom_center(l)) is not None and _cy >= (1.0 - _BOTTOM_BAND)]
    candidates_pool = bottom if bottom else lines  # fallback: usar todas

    # 2. Normalizar y recolectar candidatas individuales
    raw: list[str] = []
    for line in candidates_pool:
        clean = _normalize_mrz_text(line.text)
        if _is_mrz_candidate(clean):
            snapped = _snap_length(clean)
            if snapped:
                raw.append(snapped)

    # 3. Intentar fusionar líneas partidas (OCR divide una línea MRZ en dos)
    merged = _try_merge(raw)

    # 4. Seleccionar el conjunto que forma un formato MRZ completo
    return _pick_best(merged)


def _try_merge(lines: list[str]) -> list[str]:
    """Concatena pares de líneas cortas adyacentes que juntas forman una longitud MRZ válida."""
    result = []
    i = 0
    while i < len(lines):
        if i + 1 < len(lines):
            concat  = lines[i] + lines[i + 1]
            snapped = _snap_length(concat)
            if snapped and re.match(r'^[A-Z0-9<]+$', snapped):
                result.append(snapped)
                i += 2
                continue
        result.append(lines[i])
        i += 1
    return result


def _pick_best(candidates: list[str]) -> list[str]:
    """Devuelve las líneas que forman un formato MRZ completo (TD3/MRV-A, TD2/MRV-B o TD1)."""
    td3  = [l for l in candidates if len(l) == 44]
    td2  = [l for l in candidates if len(l) == 36]
    td1  = [l for l in candidates if len(l) == 30]

    if len(td3) >= 2:
        return td3[:2]
    if len(td2) >= 2:
        return td2[:2]
    if len(td1) >= 3:
        return td1[:3]
    return []


def _reconstruct(lines: list[str]) -> list[str]:
    if not lines:
        return lines
    target = len(lines[0])
    return [line.ljust(target, "<")[:target] for line in lines]


def _try_checker(checker_cls, mrz_string: str):
    try:
        return checker_cls(mrz_string)
    except Exception as e:
        print(f"{checker_cls.__name__} failed: {type(e).__name__}: {e}")
        return None


def _build_checker(lines: list[str]):
    line_count  = len(lines)
    line_length = len(lines[0]) if lines else 0
    mrz_string  = "\n".join(lines)

    if line_count == 2 and line_length == 44:
        from mrz.checker.mrva import MRVACodeChecker
        from mrz.checker.td3 import TD3CodeChecker
        checkers = [MRVACodeChecker, TD3CodeChecker] if lines[0].startswith("V") else [TD3CodeChecker, MRVACodeChecker]
        for cls in checkers:
            checker = _try_checker(cls, mrz_string)
            if checker is not None:
                return checker

    if line_count == 2 and line_length == 36:
        from mrz.checker.mrvb import MRVBCodeChecker
        from mrz.checker.td2 import TD2CodeChecker
        checkers = [MRVBCodeChecker, TD2CodeChecker] if lines[0].startswith("V") else [TD2CodeChecker, MRVBCodeChecker]
        for cls in checkers:
            checker = _try_checker(cls, mrz_string)
            if checker is not None:
                return checker

    if line_count == 3 and line_length == 30:
        from mrz.checker.td1 import TD1CodeChecker
        return _try_checker(TD1CodeChecker, mrz_string)

    return None


def _parse(lines: list[str]) -> MRZResult | None:
    try:
        print("MRZ lines:")
        for i, line in enumerate(lines):
            print(f"  {i + 1}: {repr(line)} len={len(line)}")

        checker = _build_checker(lines)
        if checker is None:
            print("Unsupported MRZ format")
            return None

        is_valid = bool(checker)
        print(f"Checker type: {type(checker).__name__} valid={is_valid}")

        try:
            f = checker.fields()
        except Exception as field_error:
            print(f"MRZ fields error: {type(field_error).__name__}: {field_error}")
            return MRZResult(
                valid=False, surname=None, given_names=None, country=None,
                date_of_birth=None, expiry_date=None, document_number=None, sex=None,
            )

        return MRZResult(
            valid           = is_valid,
            surname         = getattr(f, "surname",     None),
            given_names     = getattr(f, "name", getattr(f, "names", getattr(f, "given_names", None))),
            country         = getattr(f, "country",     None),
            date_of_birth   = getattr(f, "birth_date",  None),
            expiry_date     = getattr(f, "expiry_date", None),
            document_number = getattr(f, "document_number", getattr(f, "number", None)),
            sex             = getattr(f, "sex",         None),
        )

    except Exception as e:
        print(f"MRZ parse error: {type(e).__name__}: {e}")
        return None


def detect(lines: list[TextLine]) -> MRZResult | None:
    mrz_lines = _find_lines(lines)
    print(f"MRZ candidate lines found: {len(mrz_lines)}")

    if len(mrz_lines) < 2:
        return None

    mrz_lines = _reconstruct(mrz_lines)
    result    = _parse(mrz_lines)
    print(f"MRZ parse result: {result.valid if result else None}")
    return result

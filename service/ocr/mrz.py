import re

from service.logging_config import get_logger
from .models import TextLine, MRZResult


log = get_logger(__name__)

MRZ_PATTERN = re.compile(r'^[A-Z0-9<]{30,44}$')


def _normalize_mrz_text(text: str) -> str:
    return (
        text.strip()
        .replace(" ", "")
        .replace("«", "<")
        .replace("‹", "<")
        .replace("＜", "<")
        .upper()
    )


def _find_lines(lines: list[TextLine]) -> list[str]:
    candidates = []

    for line in lines:
        clean = _normalize_mrz_text(line.text)

        if MRZ_PATTERN.match(clean) and "<" in clean:
            candidates.append(clean)

    td3_or_mrva = [l for l in candidates if len(l) == 44]
    td2_or_mrvb = [l for l in candidates if len(l) == 36]
    td1 = [l for l in candidates if len(l) == 30]

    # Visa MRV-A o pasaporte TD3: 2 líneas de 44
    if len(td3_or_mrva) >= 2:
        return td3_or_mrva[:2]

    # Visa MRV-B o TD2: 2 líneas de 36
    if len(td2_or_mrvb) >= 2:
        return td2_or_mrvb[:2]

    # ID card TD1: 3 líneas de 30
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
        checker = checker_cls(mrz_string)
        return checker
    except Exception as e:
        log.debug("%s failed: %s: %s", checker_cls.__name__, type(e).__name__, e)
        return None


def _build_checker(lines: list[str]):
    line_count = len(lines)
    line_length = len(lines[0]) if lines else 0
    mrz_string = "\n".join(lines)

    if line_count == 2 and line_length == 44:
        from mrz.checker.mrva import MRVACodeChecker
        from mrz.checker.td3 import TD3CodeChecker

        # Si empieza con V, normalmente es visa MRV-A.
        checkers = [MRVACodeChecker, TD3CodeChecker] if lines[0].startswith("V") else [TD3CodeChecker, MRVACodeChecker]

        for checker_cls in checkers:
            checker = _try_checker(checker_cls, mrz_string)
            if checker is not None:
                return checker

    if line_count == 2 and line_length == 36:
        from mrz.checker.mrvb import MRVBCodeChecker
        from mrz.checker.td2 import TD2CodeChecker

        # Si empieza con V, normalmente es visa MRV-B.
        checkers = [MRVBCodeChecker, TD2CodeChecker] if lines[0].startswith("V") else [TD2CodeChecker, MRVBCodeChecker]

        for checker_cls in checkers:
            checker = _try_checker(checker_cls, mrz_string)
            if checker is not None:
                return checker

    if line_count == 3 and line_length == 30:
        from mrz.checker.td1 import TD1CodeChecker
        return _try_checker(TD1CodeChecker, mrz_string)

    return None


def _parse(lines: list[str]) -> MRZResult | None:
    try:
        log.debug("MRZ lines: %s", [(repr(l), len(l)) for l in lines])

        mrz_string = "\n".join(lines)
        log.debug("MRZ total length=%d", len(mrz_string))

        checker = _build_checker(lines)

        if checker is None:
            log.warning("unsupported MRZ format")
            return None

        is_valid = bool(checker)
        log.debug("MRZ checker=%s valid=%s", type(checker).__name__, is_valid)

        try:
            f = checker.fields()
        except Exception as field_error:
            log.warning("MRZ fields error: %s: %s", type(field_error).__name__, field_error)
            return MRZResult(
                valid=False,
                surname=None,
                given_names=None,
                country=None,
                birth_date=None,
                expiry_date=None,
                number=None,
                sex=None
            )

        return MRZResult(
            valid=is_valid,
            surname=getattr(f, "surname", None),
            given_names=getattr(f, "name", getattr(f, "names", getattr(f, "given_names", None))),
            country=getattr(f, "country", None),
            birth_date=getattr(f, "birth_date", None),
            expiry_date=getattr(f, "expiry_date", None),
            number=getattr(f, "document_number", getattr(f, "number", None)),
            sex=getattr(f, "sex", None)
        )

    except Exception as e:
        log.error("MRZ parse error: %s: %s", type(e).__name__, e)
        return None


def detect(lines: list[TextLine]) -> MRZResult | None:
    mrz_lines = _find_lines(lines)

    log.debug("MRZ candidate lines found: %d", len(mrz_lines))

    if len(mrz_lines) < 2:
        return None

    mrz_lines = _reconstruct(mrz_lines)
    result = _parse(mrz_lines)

    log.debug("MRZ parse result: valid=%s", result.valid if result else None)

    return result
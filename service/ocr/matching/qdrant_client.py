import uuid
from typing import Any, Optional


def _try_import():
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels
        return QdrantClient, qmodels
    except ImportError:
        return None, None


_CLIENT_CACHE: dict[str, Any] = {}


def _coerce_id(point_id: Any) -> Any:
    """Qdrant point IDs must be int or UUID string. Coerce arbitrary strings to a stable UUID5."""
    if isinstance(point_id, int):
        return point_id
    if isinstance(point_id, str):
        try:
            uuid.UUID(point_id)
            return point_id
        except (ValueError, AttributeError):
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id))
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(point_id)))


def is_available() -> bool:
    client_cls, _ = _try_import()
    return client_cls is not None


def get_client(url: str) -> Optional[Any]:
    """
    Accepts three forms for `url`:
      - http://host:port or https://host:port  → HTTP client (Docker / remote Qdrant)
      - ":memory:"                              → in-memory local mode (ephemeral)
      - any other string (e.g. "./qdrant_data") → local persistent file mode (no Docker)
    """
    if url in _CLIENT_CACHE:
        return _CLIENT_CACHE[url]
    client_cls, _ = _try_import()
    if client_cls is None:
        return None
    try:
        if url.startswith(("http://", "https://")):
            client = client_cls(url=url)
        elif url == ":memory:":
            client = client_cls(location=":memory:")
        else:
            client = client_cls(path=url)
        _CLIENT_CACHE[url] = client
        return client
    except Exception as e:
        print(f"[qdrant] connect failed: {type(e).__name__}: {e}")
        return None


def _build_filter(filters: dict[str, Any], qmodels) -> Optional[Any]:
    conditions = []
    for key, value in filters.items():
        if value is None:
            continue
        conditions.append(
            qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))
        )
    if not conditions:
        return None
    return qmodels.Filter(must=conditions)


def search(
    url: str,
    collection: str,
    vector: list[float],
    filters: dict[str, Any] | None = None,
    limit: int = 1,
    score_threshold: float = 0.75,
) -> list[dict]:
    """
    Returns list of hits like [{"template_id": int|str, "score": float, "payload": dict}].
    Empty list if no hit, qdrant unavailable, or any error.
    """
    client_cls, qmodels = _try_import()
    if client_cls is None:
        return []

    client = get_client(url)
    if client is None:
        return []

    try:
        q_filter = _build_filter(filters or {}, qmodels)
        response = client.query_points(
            collection_name = collection,
            query           = vector,
            query_filter    = q_filter,
            limit           = limit,
            score_threshold = score_threshold,
        )
        hits = response.points if hasattr(response, "points") else response
        return [
            {
                "template_id": (h.payload or {}).get("template_id") if h.payload else None,
                "score":       float(h.score),
                "payload":     h.payload or {},
            }
            for h in hits
        ]
    except Exception as e:
        print(f"[qdrant] search failed: {type(e).__name__}: {e}")
        return []


def upsert(
    url: str,
    collection: str,
    point_id: str,
    vector: list[float],
    payload: dict,
    vector_size: int = 1024,
) -> bool:
    client_cls, qmodels = _try_import()
    if client_cls is None:
        return False

    client = get_client(url)
    if client is None:
        return False

    try:
        existing = {c.name for c in client.get_collections().collections}
        if collection not in existing:
            client.create_collection(
                collection_name = collection,
                vectors_config  = qmodels.VectorParams(
                    size     = vector_size,
                    distance = qmodels.Distance.COSINE,
                ),
            )

        client.upsert(
            collection_name = collection,
            points          = [qmodels.PointStruct(
                id      = _coerce_id(point_id),
                vector  = vector,
                payload = payload,
            )],
        )
        return True
    except Exception as e:
        print(f"[qdrant] upsert failed: {type(e).__name__}: {e}")
        return False

"""Download model weights listed in the manifest.

Usage (with the api-doc-fraud venv, which has `liveness` installed):

    python -m service.liveness.scripts.download_weights <name>
    python -m service.liveness.scripts.download_weights --all
    python -m service.liveness.scripts.download_weights --list

Prints each file's SHA256 to pin in the manifest; the first download
establishes the canonical hash, later ones verified via the loader. When
an entry has `archive_member`, the URL is a .zip and that member is
extracted (the pinned hash is the member's, not the zip's).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile

from service.liveness.infrastructure.manifest import MANIFEST, WeightEntry


def _sha256_stream(reader, chunk_size: int = 1 << 20) -> tuple[bytes, str]:
    hasher = hashlib.sha256()
    data = bytearray()
    while True:
        chunk = reader.read(chunk_size)
        if not chunk:
            break
        hasher.update(chunk)
        data.extend(chunk)
    return bytes(data), hasher.hexdigest()


def download(entry: WeightEntry) -> int:
    target = entry.absolute_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"[skip] {entry.name}: already at {target}")
        return 0
    print(f"[get ] {entry.name}: {entry.url}")
    try:
        with urllib.request.urlopen(entry.url) as response:
            raw, raw_sha = _sha256_stream(response)
    except Exception as exc:  # noqa: BLE001
        print(f"[fail] {entry.name}: {exc}", file=sys.stderr)
        return 1

    if entry.archive_member:
        # The URL is a zip; extract the named member and hash THAT.
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                data = zf.read(entry.archive_member)
        except (KeyError, zipfile.BadZipFile) as exc:
            print(f"[fail] {entry.name}: extracting {entry.archive_member!r}: {exc}",
                  file=sys.stderr)
            return 1
        sha = hashlib.sha256(data).hexdigest()
        print(f"[zip ] {entry.name}: extracted {entry.archive_member} "
              f"({len(data):,} bytes) from {len(raw):,}-byte archive")
    else:
        data, sha = raw, raw_sha

    target.write_bytes(data)
    print(f"[ok  ] {entry.name}: wrote {len(data):,} bytes")
    print(f"       sha256={sha}")
    print(f"       path={target}")
    if entry.sha256 is not None and sha != entry.sha256:
        print(
            f"[warn] sha256 mismatch — manifest expects {entry.sha256}",
            file=sys.stderr,
        )
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download liveness model weights.")
    parser.add_argument("name", nargs="?", help="Weight name from manifest.")
    parser.add_argument("--all", action="store_true", help="Download every entry.")
    parser.add_argument("--list", action="store_true", help="List manifest entries and exit.")
    args = parser.parse_args(argv)

    if args.list:
        for name, entry in MANIFEST.items():
            print(f"  {name:<32} {entry.url}")
        return 0

    if args.all:
        rc = 0
        for entry in MANIFEST.values():
            rc |= download(entry)
        return rc

    if not args.name:
        parser.error("Provide a weight name, --all, or --list.")
    if args.name not in MANIFEST:
        print(f"Unknown weight: {args.name}", file=sys.stderr)
        return 1
    return download(MANIFEST[args.name])


if __name__ == "__main__":
    sys.exit(main())

"""Verify all files in a delivery ZIP against its embedded SHA256SUMS.txt."""

import hashlib
import sys
from pathlib import Path
from zipfile import ZipFile


def main() -> None:
    archive = Path(sys.argv[1])
    with ZipFile(archive) as bundle:
        entries = bundle.namelist()
        manifest = bundle.read("SHA256SUMS.txt").decode("utf-8-sig")
        expected = {}
        for line in manifest.splitlines():
            digest, name = line.split("  ", 1)
            expected[name] = digest
        actual = {name for name in entries if not name.endswith("/")}
        if actual != set(expected) | {"SHA256SUMS.txt"}:
            raise SystemExit("Archive file list differs from SHA256SUMS.txt")
        if len(actual) != len([name for name in entries if not name.endswith("/")]):
            raise SystemExit("Archive contains duplicate entries")
        for name, digest in expected.items():
            with bundle.open(name) as stream:
                computed = hashlib.file_digest(stream, "sha256").hexdigest()
            if computed != digest:
                raise SystemExit(f"Checksum mismatch: {name}")
        print(f"Verified {len(expected)} files in {archive.name}")


if __name__ == "__main__":
    main()

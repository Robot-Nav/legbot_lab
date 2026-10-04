#!/usr/bin/env python3
"""Create a deterministic transfer archive for the KickPi."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import tarfile
from pathlib import Path


EXCLUDED_PARTS = {"build", "__pycache__", ".git"}
EXCLUDED_SUFFIXES = {".pyc"}


def included(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return (
        not any(part in EXCLUDED_PARTS for part in relative.parts)
        and (not relative.parts or relative.parts[0] != root.name)
        and path.suffix not in EXCLUDED_SUFFIXES
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="w1w_runtime.tar.gz")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and included(path, root) and path.resolve() != output
    )
    with output.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="", fileobj=raw_output, mode="wb", compresslevel=9, mtime=0
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in files:
                    arcname = Path("w1w_runtime") / path.relative_to(root)
                    executable = (
                        path.parent.name == "scripts"
                        or (path.suffix == ".py" and path.read_bytes().startswith(b"#!"))
                    )

                    def normalize(info: tarfile.TarInfo, executable=executable) -> tarfile.TarInfo:
                        info.uid = 0
                        info.gid = 0
                        info.uname = "root"
                        info.gname = "root"
                        info.mtime = 0
                        info.mode = 0o755 if executable else 0o644
                        return info

                    archive.add(path, arcname=str(arcname), recursive=False, filter=normalize)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"created {output}")
    print(f"files={len(files)} bytes={output.stat().st_size} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import gzip
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        return int(response.headers["Content-Length"])


def download_part(url: str, path: Path, start: int, end: int) -> None:
    expected = end - start + 1
    if path.exists() and path.stat().st_size > expected:
        path.unlink()
    consecutive_failures = 0
    while not path.exists() or path.stat().st_size < expected:
        have = path.stat().st_size if path.exists() else 0
        request_start = start + have
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={request_start}-{end}",
                    "User-Agent": "psoriasis-feasibility/1.0",
                },
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(f"Server returned HTTP {response.status}, expected 206")
                with path.open("ab") as output:
                    while block := response.read(64 * 1024):
                        remaining = expected - output.tell()
                        output.write(block[:remaining])
                        if output.tell() == expected:
                            break
            consecutive_failures = 0
        except Exception:
            consecutive_failures += 1
            if consecutive_failures >= 20:
                raise
            time.sleep(min(2 ** min(consecutive_failures, 4), 20))
    if path.stat().st_size != expected:
        raise RuntimeError(f"Part has {path.stat().st_size} bytes, expected {expected}")


def download(url: str, target: Path, workers: int) -> None:
    size = remote_size(url)
    part_dir = target.parent / f".{target.name}.parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = (size + workers - 1) // workers
    jobs = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index in range(workers):
            start = index * chunk_size
            end = min(size - 1, (index + 1) * chunk_size - 1)
            if start > end:
                continue
            part = part_dir / f"{index:03d}.part"
            jobs.append((index, part, pool.submit(download_part, url, part, start, end)))
        for index, _, future in jobs:
            future.result()
            print(f"part {index + 1}/{len(jobs)} complete", flush=True)

    temporary = target.with_suffix(target.suffix + ".complete")
    with temporary.open("wb") as output:
        for _, part, _ in jobs:
            with part.open("rb") as source:
                while block := source.read(1024 * 1024):
                    output.write(block)
    if temporary.stat().st_size != size:
        raise RuntimeError("Assembled file size does not match the server")
    if target.suffix == ".gz":
        with gzip.open(temporary, "rb") as handle:
            while handle.read(1024 * 1024):
                pass
    os.replace(temporary, target)
    for _, part, _ in jobs:
        part.unlink()
    part_dir.rmdir()
    print(f"verified {target} ({size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("target", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    download(args.url, args.target, args.workers)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download and extract the ten frozen GSE202011 H5 matrices from GEO.

The coordinate map is versioned in scripts/pri132_revision and is not inferred
from pathology images. The GEO archive is public but not redistributed here.
"""
from __future__ import annotations
import argparse, hashlib, tarfile, urllib.request
from pathlib import Path

URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE202nnn/GSE202011/suppl/GSE202011_RAW.tar"
EXPECTED_SHA256 = "76ad74e0c38fefbebfdd4fb2e92cf2a3343e87c8cba322a7fe76abec25b6e5a0"
GSM = {"GSM6086615", "GSM6086616", "GSM6086617", "GSM6086619", "GSM6086625", "GSM6086626", "GSM6086630", "GSM6086631", "GSM6086632", "GSM6086633"}

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1<<20), b""): h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-dir", type=Path, default=Path("data/raw/pri_1_32_gse202011")); args=p.parse_args()
    out=args.data_dir; h5=out/"h5"; h5.mkdir(parents=True, exist_ok=True); archive=out/"GSE202011_RAW.tar"
    if not archive.exists(): urllib.request.urlretrieve(URL, archive)
    actual=digest(archive)
    if actual != EXPECTED_SHA256: raise RuntimeError(f"GEO archive SHA-256 mismatch: {actual}")
    with tarfile.open(archive) as tf:
        members=[m for m in tf.getmembers() if m.isfile() and m.name.endswith("filtered_feature_bc_matrix.h5") and any(g in m.name for g in GSM)]
        if len(members) != 10: raise RuntimeError(f"Expected 10 frozen H5 files, found {len(members)}")
        for m in members:
            target=h5/Path(m.name).name
            if not target.exists():
                source=tf.extractfile(m); target.write_bytes(source.read())
    print(f"Prepared {len(list(h5.glob('*_filtered_feature_bc_matrix.h5')))} matrices in {h5}")
if __name__ == "__main__": main()

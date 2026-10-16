"""Download and verify the public GEO inputs required by the v24 correction."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/pri_audit_v24_gse151177/reproducibility_manifest.json"
USER_AGENT = "PRI-Audit reproducibility fetcher/1.0 (public GEO data)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_url(relative_path: str) -> str:
    name = Path(relative_path).name
    if name == "GSE151177_series_matrix.txt.gz":
        return "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE151nnn/GSE151177/matrix/GSE151177_series_matrix.txt.gz"
    gsm = name.split("_", 1)[0]
    if not gsm.startswith("GSM") or not gsm[3:].isdigit():
        raise ValueError(f"Unexpected GEO input name: {name}")
    geo_bucket = gsm[:7] + "nnn"
    return f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{geo_bucket}/{gsm}/suppl/{name}"


def valid(path: Path, expected_bytes: int, expected_sha: str) -> bool:
    return path.is_file() and path.stat().st_size == expected_bytes and sha256(path) == expected_sha


def download(entry: dict[str, object]) -> None:
    relative_path = str(entry["path"])
    target = (ROOT / relative_path).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Input path escapes repository: {relative_path}")
    expected_bytes = int(entry["bytes"])
    expected_sha = str(entry["sha256"])
    if valid(target, expected_bytes, expected_sha):
        print(f"verified: {relative_path}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".download")
    url = source_url(relative_path)
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
            with urlopen(request, timeout=180) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if not valid(temporary, expected_bytes, expected_sha):
                raise RuntimeError(f"Downloaded file failed size or SHA-256 check: {relative_path}")
            temporary.replace(target)
            print(f"downloaded and verified: {relative_path}")
            return
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < 4:
                time.sleep(attempt * 2)
    raise RuntimeError(f"Could not retrieve {relative_path} from {url}") from last_error


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise RuntimeError(f"No input records in {MANIFEST}")
    for entry in inputs:
        download(entry)
    print(f"Verified all {len(inputs)} public GEO inputs against the v24 manifest.")


if __name__ == "__main__":
    main()

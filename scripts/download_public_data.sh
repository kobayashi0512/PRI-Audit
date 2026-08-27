#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${ROOT_DIR}/data/raw"
mkdir -p "${RAW_DIR}"

resume_gzip() {
  local target="$1"
  local url="$2"
  local attempt
  for attempt in $(seq 1 40); do
    if [[ -f "${target}" ]] && gzip -t "${target}" 2>/dev/null; then
      echo "verified: ${target}"
      return 0
    fi
    echo "download attempt ${attempt}: ${target}"
    curl --http1.1 -sS -L -C - \
      --retry 2 --retry-all-errors --connect-timeout 20 --max-time 180 \
      -o "${target}" "${url}" || true
  done
  gzip -t "${target}"
}

resume_plain() {
  local target="$1"
  local url="$2"
  if [[ -s "${target}" ]]; then
    echo "present: ${target}"
    return 0
  fi
  curl --http1.1 -sS -L --retry 5 --retry-all-errors \
    --connect-timeout 20 -o "${target}" "${url}"
}

resume_gzip \
  "${RAW_DIR}/GSE206326_GAS_Blood_Teff_RNA_counts.txt.gz" \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE206nnn/GSE206326/suppl/GSE206326_GAS_Blood_Teff_RNA_counts.txt.gz"
resume_gzip \
  "${RAW_DIR}/GSE206326_Blood.matrix.data.upload.txt.gz" \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE206nnn/GSE206326/suppl/GSE206326_Blood.matrix.data.upload.txt.gz"
resume_gzip \
  "${RAW_DIR}/GSE74235_GAS_HCQ_GenewiseCounts.txt.gz" \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE74nnn/GSE74235/suppl/GSE74235_GAS_HCQ_GenewiseCounts.txt.gz"
resume_gzip \
  "${RAW_DIR}/GSE24767_series_matrix.txt.gz" \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE24nnn/GSE24767/matrix/GSE24767_series_matrix.txt.gz"
resume_gzip \
  "${RAW_DIR}/GPL6947.annot.gz" \
  "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL6nnn/GPL6947/annot/GPL6947.annot.gz"
resume_gzip \
  "${RAW_DIR}/GSE121212_readcount.txt.gz" \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE121nnn/GSE121212/suppl/GSE121212_readcount.txt.gz"
resume_gzip \
  "${RAW_DIR}/GSE121212_series_matrix.txt.gz" \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE121nnn/GSE121212/matrix/GSE121212_series_matrix.txt.gz"
resume_plain \
  "${RAW_DIR}/hgnc_complete_set.txt" \
  "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
resume_gzip \
  "${RAW_DIR}/GSE57225_series_matrix.txt.gz" \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE57nnn/GSE57225/matrix/GSE57225_series_matrix.txt.gz"
resume_plain \
  "${RAW_DIR}/GPL14550_full.txt" \
  "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL14550&targ=self&form=text&view=data"

echo "All requested public files downloaded and verified."

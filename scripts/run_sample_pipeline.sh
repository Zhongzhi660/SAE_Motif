#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

ATTACK_TAG="sourceA_targetA"
NON_ATTACK_TAG="non_attack_reference"
THR="2.0"

ATTACK_PROC_DIR="${ROOT}/data/processed/${ATTACK_TAG}"
ATTACK_SAE_DIR="${ROOT}/data/sae/${ATTACK_TAG}/threshold_${THR}"
NON_ATTACK_SAE_DIR="${ROOT}/data/sae/${NON_ATTACK_TAG}/threshold_${THR}"
OUT_MOTIF="${ROOT}/outputs/sample_motif_structure"
OUT_CROSS="${ROOT}/outputs/sample_cross_method"

echo "== Sample Pipeline =="
echo "Repository root: ${ROOT}"
echo "Attack tag:      ${ATTACK_TAG}"
echo "Non-attack tag:  ${NON_ATTACK_TAG}"
echo "Threshold:       ${THR}"
echo

echo "== Step 0: Preflight check =="
python "${ROOT}/scripts/check_setup.py" || true
echo

echo "== Step 1: Show bundled sample inputs =="
ls -1 "${ATTACK_PROC_DIR}"
ls -1 "${ATTACK_SAE_DIR}"
ls -1 "${NON_ATTACK_SAE_DIR}"
echo

echo "== Step 2: Span grouping demo =="
echo "This re-runs grouping on the toy textspans file and rewrites features.tsv."
python -m src.annotations.groupby_textspans "${ATTACK_TAG}" "${THR}" llama
echo

echo "== Step 3: Optional feature explanation demo =="
if [[ -n "${API_KEY:-}" ]]; then
  echo "API_KEY detected. Re-running feature explanation on the toy features.tsv."
  python -m src.annotations.annotate_explanations \
    "${ATTACK_SAE_DIR}/features.tsv" \
    "${ATTACK_SAE_DIR}/features_explained.tsv"
else
  echo "Skipping annotation because API_KEY is not set."
  echo "Bundled toy file remains available at:"
  echo "  ${ATTACK_SAE_DIR}/features_explained.tsv"
fi
echo

echo "== Step 4: Core motif-structure analysis =="
mkdir -p "${OUT_MOTIF}"
python -m src.analysis.motif_structure_analysis \
  --atk-base "${ROOT}/data/sae/${ATTACK_TAG}" \
  --non-attack-base "${ROOT}/data/sae/${NON_ATTACK_TAG}" \
  --save-dir "${OUT_MOTIF}"
echo

echo "== Step 5: Core cross-method analysis =="
mkdir -p "${OUT_CROSS}"
python -m src.analysis.cross_method_analysis \
  --atk-base "${ROOT}/data/sae/${ATTACK_TAG}" \
  --non-attack-base "${ROOT}/data/sae/${NON_ATTACK_TAG}" \
  --labels "${ATTACK_PROC_DIR}/labels.tsv" \
  --threshold "${THR}" \
  --k 3 \
  --top-k 2 \
  --n-perm 10 \
  --n-boot 20 \
  --feature-rate-thr 0.0 \
  --feature-global-rate-thr 0.0 \
  --top-texts-per-feature 2 \
  --features-explained-tsv "${ATTACK_SAE_DIR}/features_explained.tsv" \
  --save-dir "${OUT_CROSS}"
echo

echo "== Done =="
echo "Outputs written to:"
echo "  ${OUT_MOTIF}"
echo "  ${OUT_CROSS}"

#!/usr/bin/env bash
# 検証に必要なファイルだけを Brev へ送る（third_party と .venv は送らない）
set -euo pipefail
cd "$(dirname "$0")/../.."

WORKSPACE="${REUSE_BREV_WORKSPACE:-cm-nakamura-re-use}"
REMOTE_DIR="${REUSE_REMOTE_DIR:-/home/ubuntu/re-use}"

# 送るものを一時ディレクトリにまとめ、ディレクトリごと 1 回で送る
STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT
mkdir -p "${STAGE}/work"
cp -R scripts "${STAGE}/"
[[ -d work/inputs ]] && cp -R work/inputs "${STAGE}/work/"

brev exec "${WORKSPACE}" "mkdir -p ${REMOTE_DIR}"
brev copy "${STAGE}/" "${WORKSPACE}:${REMOTE_DIR}/"

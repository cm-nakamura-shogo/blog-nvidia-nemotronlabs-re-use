#!/usr/bin/env bash
# 検証に必要なファイルだけを Brev へ送る（third_party と .venv は送らない）
# brev exec / brev copy は多重化接続が詰まると戻らないことがあったため、ssh と rsync を直接使う
set -euo pipefail
cd "$(dirname "$0")/../.."

WORKSPACE="${REUSE_BREV_WORKSPACE:-cm-nakamura-re-use}"
REMOTE_DIR="${REUSE_REMOTE_DIR:-/home/ubuntu/re-use}"
SSH=(ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)

"${SSH[@]}" "${WORKSPACE}" "mkdir -p ${REMOTE_DIR}/work"
rsync -az -e "${SSH[*]}" scripts "${WORKSPACE}:${REMOTE_DIR}/"
rsync -az -e "${SSH[*]}" work/inputs "${WORKSPACE}:${REMOTE_DIR}/work/"

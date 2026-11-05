#!/usr/bin/env bash
# Brev 上でコマンドを実行する、または結果を取得する
#   bash scripts/host/remote.sh run "<command>"       # ~/re-use で GPU 環境を有効にして実行
#   bash scripts/host/remote.sh fetch <remote_path>... # work/torch/ などへ同じ相対パスで取得
set -euo pipefail
cd "$(dirname "$0")/../.."

WORKSPACE="${REUSE_BREV_WORKSPACE:-cm-nakamura-re-use}"
REMOTE_DIR="${REUSE_REMOTE_DIR:-/home/ubuntu/re-use}"
SSH=(ssh -T -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)

case "$1" in
  run)
    # セットアップ前は仮想環境がないので、あるときだけ有効にする
    "${SSH[@]}" "${WORKSPACE}" "cd ${REMOTE_DIR} && { [ ! -f work/.venv-gpu/bin/activate ] || source work/.venv-gpu/bin/activate; } && $2"
    ;;
  fetch)
    shift
    for rel in "$@"; do
      mkdir -p "$(dirname "${rel}")"
      rsync -az -e "${SSH[*]}" "${WORKSPACE}:${REMOTE_DIR}/${rel}" "$(dirname "${rel}")/"
    done
    ;;
esac

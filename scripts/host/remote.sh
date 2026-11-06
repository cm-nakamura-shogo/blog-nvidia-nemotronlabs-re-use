#!/usr/bin/env bash
# Brev 上でコマンドを実行する、または結果を取得する
#   bash scripts/host/remote.sh run "<command>"       # ~/re-use で GPU 環境を有効にして実行
#   bash scripts/host/remote.sh fetch <remote_path>... # 同じ相対パスで手元に取得
set -euo pipefail
cd "$(dirname "$0")/../.."

WORKSPACE="${REUSE_BREV_WORKSPACE:-cm-nakamura-re-use}"
REMOTE_DIR="${REUSE_REMOTE_DIR:-/home/ubuntu/re-use}"

case "$1" in
  run)
    # セットアップ前は仮想環境がないので、あるときだけ有効にする
    brev exec "${WORKSPACE}" "cd ${REMOTE_DIR} && { [ ! -f work/.venv-gpu/bin/activate ] || source work/.venv-gpu/bin/activate; } && $2"
    ;;
  fetch)
    # Brev 側で 1 つの tar にまとめてから取得し、手元で展開する
    shift
    archive="/tmp/re-use-fetch.tgz"
    brev exec "${WORKSPACE}" "cd ${REMOTE_DIR} && tar czf ${archive} $*"
    brev copy "${WORKSPACE}:${archive}" work/fetch.tgz
    tar xzf work/fetch.tgz
    rm work/fetch.tgz
    ;;
esac

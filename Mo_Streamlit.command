#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if ! .venv312/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12)' 2>/dev/null; then
    if command -v python3.12 >/dev/null 2>&1; then
        python3.12 -m venv .venv312
    elif [[ -x /opt/homebrew/opt/python@3.12/bin/python3.12 ]]; then
        /opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv312
    else
        printf 'Cần Python 3.12. Chạy brew install python@3.12 rồi mở lại file này.\n' >&2
        exit 1
    fi
fi

if ! .venv312/bin/python -c 'import streamlit, pandas, altair, numpy, PIL' 2>/dev/null; then
    printf 'Đang cài thư viện Streamlit…\n'
    .venv312/bin/python -m pip install -r requirements-streamlit.txt
fi

copy_ckpt() {
    local src="$1" dst="$2"
    if [[ ! -f "$dst" && -f "$src" ]]; then
        mkdir -p "$(dirname "$dst")"
        cp "$src" "$dst"
    fi
}
copy_ckpt resnet50_best.pth models/resnet50/model.pth
copy_ckpt efficientnet_b0_best.pth models/efficientnet_b0/model.pth
copy_ckpt swin_tiny_best.pth models/swin/model.pth

export NEUROSCAN_DEVICE="${NEUROSCAN_DEVICE:-cpu}"
shopt -s nullglob
pth_files=(models/*/model.pth)
shopt -u nullglob
if (( ${#pth_files[@]} > 0 )); then
    if ! .venv312/bin/python -c 'import torch, torchvision, timm' 2>/dev/null; then
        .venv312/bin/python -m pip install -r requirements-pytorch.txt
    fi
    .venv312/bin/python app.py --verify-ready
fi

printf '\nĐang mở NeuroScan Streamlit tại http://127.0.0.1:8501\n'
exec .venv312/bin/python -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501

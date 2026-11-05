#!/bin/bash
set -e
cd "$(dirname "$0")"

if ! .venv312/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12)' 2>/dev/null; then
    if command -v python3.12 >/dev/null 2>&1; then
        python3.12 -m venv .venv312
    elif [[ -x /opt/homebrew/opt/python@3.12/bin/python3.12 ]]; then
        /opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv312
    else
        printf 'Cần Python 3.12. Trong Terminal, chạy brew install python@3.12 rồi mở lại file này.\n' >&2
        exit 1
    fi
fi

if ! .venv312/bin/python -c 'import fastapi, uvicorn, numpy, PIL, torch, torchvision' 2>/dev/null; then
    printf 'Đang cài thư viện NeuroScan cho lần chạy đầu tiên…\n'
    .venv312/bin/python -m pip install -r requirements-web.txt -r requirements-pytorch.txt
fi

# Mặc định dùng CPU cho lần chạy đầu tiên trên Mac.
export NEUROSCAN_DEVICE="${NEUROSCAN_DEVICE:-cpu}"
.venv312/bin/python app.py --verify-models
printf '\nKhi thấy Uvicorn running, mở http://127.0.0.1:8002 trong trình duyệt.\n'
exec .venv312/bin/python -m uvicorn server:api --host 127.0.0.1 --port 8002

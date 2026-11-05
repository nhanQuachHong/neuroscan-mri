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

if ! .venv312/bin/python -c 'import fastapi, uvicorn, numpy, PIL' 2>/dev/null; then
    printf 'Đang cài thư viện giao diện…\n'
    .venv312/bin/python -m pip install -r requirements-web.txt
fi

# Chấp nhận checkpoint được tải về cạnh file mở ứng dụng.
if [[ ! -f models/resnet50/model.pth && -f resnet50_best.pth ]]; then
    mkdir -p models/resnet50
    cp resnet50_best.pth models/resnet50/model.pth
fi

export NEUROSCAN_DEVICE="${NEUROSCAN_DEVICE:-cpu}"
if [[ -f models/resnet50/model.pth ]]; then
    if ! .venv312/bin/python -c 'import torch, torchvision' 2>/dev/null; then
        .venv312/bin/python -m pip install -r requirements-pytorch.txt
    fi
    # Có checkpoint nhưng nạp lỗi: dừng và giữ nguyên thông báo kiểm tra.
    .venv312/bin/python app.py --verify-models
else
    printf '\nChưa có checkpoint ResNet50. Giao diện vẫn mở ở chế độ minh họa.\n'
    printf 'Đặt resnet50_best.pth cạnh file này rồi mở lại để bật mô hình thật.\n'
fi

printf '\nKhi thấy Uvicorn running, mở http://127.0.0.1:8002\n'
exec .venv312/bin/python -m uvicorn server:api --host 127.0.0.1 --port 8002

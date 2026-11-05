#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -e .git ]]; then
    printf 'Thư mục đã có .git. Không thay đổi lịch sử hoặc cấu hình hiện có.\n'
    exit 1
fi
git init -b feat/mri-demo-application
git remote add origin https://github.com/nhanQuachHong/neuroscan-mri.git
git fetch --no-tags origin feat/mri-demo-application
# Chỉ đặt HEAD và index; giữ nguyên mọi file làm việc đang có.
git reset --mixed origin/feat/mri-demo-application
git branch --set-upstream-to=origin/feat/mri-demo-application
git status --short --branch
printf '\nĐã khôi phục Git. Các thay đổi cục bộ vẫn được giữ nguyên.\n'

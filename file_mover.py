#!/usr/bin/env python3
import shutil
from pathlib import Path
from collections import Counter
import sys

# 검색을 시작할 루트 디렉터리
SRC_ROOT = Path("experiments/020_revision/triplet_results")

# 바뀐 파일을 옮길 대상 디렉터리
DEST_DIR = Path("experiments/020_revision/triplet_results/offline/apigraph/LCKLD-only")

#*11_1_[11~16] 부터 다시 시작 !!!! 한개도 안한 깨끗한 상태!
TARGET_CORE = "11_1_1"
REPLACEMENT = "1"
print("시작")

DEST_DIR.mkdir(parents=True, exist_ok=True)

# 1단계: 후보 파일 및 변경 후 경로 전부 수집
candidates = []  # (src_path, dest_path)
for path in SRC_ROOT.rglob("*"):
    if not path.is_file():
        continue

    name = path.name

    # "11_1_1." 또는 "11_1_1_" 를 포함하는 파일만 대상
    if TARGET_CORE+"." in name or TARGET_CORE+"_" in name:
        # 파일명에서 '11_1_1' 부분을 '1'로 치환
        new_name = name.replace(TARGET_CORE, REPLACEMENT)
        dest_path = DEST_DIR / new_name
        candidates.append((path, dest_path))

print(f"이름 변경 대상 파일 수: {len(candidates)}개")

# 2단계: 충돌 검사
conflicts_existing = [(src, dst) for (src, dst) in candidates if dst.exists()]

name_counts = Counter(dst.name for (_, dst) in candidates)
conflicts_internal = [name for (name, c) in name_counts.items() if c > 1]

if conflicts_existing or conflicts_internal:
    print("===== 이동 작업 중단 =====")
    if conflicts_existing:
        print("목적 폴더에 이미 같은 이름의 파일이 존재하는 경우:")
        for src, dst in conflicts_existing:
            print(f"  - {dst} (from {src})")

    if conflicts_internal:
        print("변경 후 파일명끼리 서로 중복되는 경우:")
        for name in conflicts_internal:
            print(f"  - 중복 파일명: {name}")

    print("위 충돌 때문에 어떤 파일도 이동하지 않았습니다.")
    sys.exit(1)

# 3단계: 실제 이동 (충돌 없을 때만)
cnt_moved = 0
for src, dst in candidates:
    shutil.move(str(src), str(dst))
    print(f"Moved: {src} -> {dst}")
    cnt_moved += 1

print("================================")
print(f"이동된 파일은 총 {cnt_moved}개 입니다.")

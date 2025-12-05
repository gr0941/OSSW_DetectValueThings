# analyze_damage_image.py
"""
단일 이미지 분석 스크립트
- 이미지 분류(Transformers image-classification pipeline, Top-1)
- 파손 정도 분석(OpenCV Laplacian variance 기반 Damage Score)
- 이미지 인자는 optional:
    - 인자를 주면 해당 경로 사용
    - 인자를 안 주면 스크립트 폴더에서 첫 번째 이미지 파일 자동 선택
출력: JSON (label, percent, damage_score, damage_level)
"""

import os
import argparse
import json

import cv2
import numpy as np
from transformers import pipeline

# 스크립트 위치로 작업 디렉토리 변경 (IDLE 등에서 실행할 때 안전)
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir:
        os.chdir(script_dir)
except NameError:
    # __file__이 없는 환경(IDLE 등)에서는 무시
    pass


def find_first_image_in_cwd():
    """
    현재 작업 폴더에서 확장자가 이미지인 (jpg/jpeg/png/bmp/webp)
    파일을 찾아서 가장 먼저 나오는 파일명을 반환.
    없으면 None.
    """
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    files = sorted(os.listdir("."))
    for f in files:
        name, ext = os.path.splitext(f)
        if ext.lower() in exts:
            return f
    return None


# ------------------------------
# Damage 분석 관련 함수들
# ------------------------------
def compute_damage_score(image, blur_size=3):
    """
    파손 정도를 나타내는 Damage Score 계산.
    Laplacian variance를 기반으로 함.
    - 값이 높을수록 표면 불규칙(스크래치/깨짐) 정도가 큼.
    - 반환값: 0.0 ~ 1.0 사이 float
    """
    # BGR → Gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 노이즈 제거용 블러
    gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    # Laplacian으로 엣지/노이즈 강도 측정
    lap = cv2.Laplacian(gray, cv2.CV_64F)

    # 분산(variance)이 높으면 손상도가 큼
    variance = lap.var()

    # 0~1 스케일 정규화 (대략적인 값)
    # 일반적으로 1000 이상이면 매우 거친 패턴 → 강한 손상
    score = min(variance / 1000.0, 1.0)

    return float(score)


def classify_damage_level(score):
    """
    damage score 기반 손상 레벨 분류
    - 입력: 0.0 ~ 1.0
    - 출력: 한글 설명 문자열
    """
    if score < 0.15:
        return "손상 없음 (no damage)"
    elif score < 0.35:
        return "경미한 손상 (light damage)"
    elif score < 0.65:
        return "중간 손상 (moderate damage)"
    else:
        return "심각한 손상 (heavy damage)"


# ------------------------------
# 메인 로직
# ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="단일 이미지 분류 + 파손 정도 분석 스크립트"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="이미지 파일 경로 (optional, 없으면 현재 폴더 첫 번째 이미지 자동 선택)",
    )
    parser.add_argument(
        "--model",
        default="google/vit-base-patch16-224",
        help="HuggingFace 이미지 분류 모델 이름",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="-1: CPU, 0: GPU0, 1: GPU1, ...",
    )
    parser.add_argument(
        "--blur-size",
        type=int,
        default=3,
        help="Damage Score 계산 시 GaussianBlur 커널 크기(홀수). 기본값=3",
    )
    args = parser.parse_args()

    # 이미지 경로 결정
    image_path = args.image
    if image_path is None:
        image_path = find_first_image_in_cwd()
        if image_path is None:
            print("오류: 이미지 인자를 주지 않았고, 현재 폴더에 (jpg/png/bmp/webp) 이미지가 없습니다.")
            return
        print(f"이미지 인자 없음 → '{image_path}' 자동 사용합니다.")

    if not os.path.exists(image_path):
        print(f"오류: 이미지 파일을 찾을 수 없습니다: {image_path}")
        return

    # --------------------------
    # 1) 파손 정도 분석 (OpenCV)
    # --------------------------
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        print(f"오류: OpenCV로 이미지를 읽을 수 없습니다: {image_path}")
        return

    damage_score = compute_damage_score(image_cv, blur_size=args.blur_size)
    damage_level = classify_damage_level(damage_score)

    # --------------------------
    # 2) 이미지 분류 (HF pipeline)
    # --------------------------
    device = args.device  # -1 for CPU, 0~ for GPU
    clf = pipeline("image-classification", model=args.model, device=device)

    # pipeline은 파일 경로나 PIL.Image를 받음; 여기선 경로 그대로 전달
    preds = clf(image_path, top_k=1)

    # preds 예시: [{'label': 'coffee mug', 'score': 0.9123}]
    label = None
    label_score = 0.0
    if isinstance(preds, list) and len(preds) > 0:
        p = preds[0]
        label = p.get("label")
        label_score = float(p.get("score", 0.0))
    else:
        print("경고: 분류 결과가 없습니다.")
        label = None
        label_score = 0.0

    # --------------------------
    # 3) 최종 결과 JSON 출력
    # --------------------------
    out = {
        "image_path": image_path,
        "label": label,
        "percent": round(label_score * 100, 2),
        "damage_score": round(damage_score, 4),
        "damage_level": damage_level,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
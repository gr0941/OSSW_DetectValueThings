# analyze_single_image.py
"""
Simple single-image analyzer (same-folder 기본)
- image 인자 optional
- 인자 없으면 스크립트 폴더에서 첫 번째 이미지 파일 사용
- HF image-classification pipeline으로 Top-1 label + percent 출력
"""

import os
import argparse
import json
from PIL import Image
from transformers import pipeline

# 스크립트 위치로 작업 디렉토리 변경 (IDLE 등에서 실행할 때 안전)
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir:
        os.chdir(script_dir)
except NameError:
    pass

def find_first_image_in_cwd():
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    files = sorted(os.listdir("."))
    for f in files:
        name, ext = os.path.splitext(f)
        if ext.lower() in exts:
            return f
    return None

def main():
    parser = argparse.ArgumentParser(description="Analyze a single image in this folder (or given path).")
    parser.add_argument("image", nargs="?", default=None, help="이미지 파일 경로 (optional)")
    parser.add_argument("--model", default="google/vit-base-patch16-224", help="HuggingFace 이미지 분류 모델 이름")
    parser.add_argument("--device", type=int, default=-1, help="-1 CPU, 0 GPU0, ...")
    args = parser.parse_args()

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

    # load pipeline
    device = args.device  # -1 for CPU
    clf = pipeline("image-classification", model=args.model, device=device)

    # pipeline은 파일 경로나 PIL.Image 받음; 여기선 경로 그대로 전달
    preds = clf(image_path, top_k=1)

    # preds 예시: [{'label': 'coffee mug', 'score': 0.9123}]
    if isinstance(preds, list) and len(preds) > 0:
        p = preds[0]
        label = p.get("label")
        score = float(p.get("score", 0.0))
        out = {"label": label, "percent": round(score * 100, 2)}
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print("분류 결과가 없습니다.")

if __name__ == "__main__":
    main()

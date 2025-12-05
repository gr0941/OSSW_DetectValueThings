# damage_detector.py
import cv2
import numpy as np

def compute_damage_score(image, blur_size=3):
    """
    파손 정도를 나타내는 Damage Score 계산.
    Laplacian variance를 기반으로 함.
    - 값이 높을수록 표면 불규칙(스크래치/깨짐) 정도가 큼.
    """
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
    """
    if score < 0.15:
        return "손상 없음 (no damage)"
    elif score < 0.35:
        return "경미한 손상 (light damage)"
    elif score < 0.65:
        return "중간 손상 (moderate damage)"
    else:
        return "심각한 손상 (heavy damage)"

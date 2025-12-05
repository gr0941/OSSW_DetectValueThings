print("========================================")
print("        파손 정도 확인 프로그램")
print("========================================\n")

while True:
    value = input("0~1 사이의 파손 지표값을 입력하세요 (종료: -1)\n")

    # 종료 조건
    if value.strip() == "-1":
        print("\n프로그램을 종료합니다. 감사합니다.")
        break

    # 입력값 확인
    try:
        score = float(value)
    except ValueError:
        print("숫자를 입력해주세요.\n")
        continue

    # 범위 검사
    if not (0 <= score <= 1):
        print("0~1 사이의 값만 입력 가능합니다.\n")
        continue

    # 파손 판정(0.5 기준)
    status = "positive" if score >= 0.5 else "negative"

    # 결과 출력
    print("\n판정 결과:")
    print(f"   상태: {status}")
    print(f"   입력값: {score:.2f}")
    print("----------------------------------------\n")
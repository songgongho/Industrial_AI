from pathlib import Path
import argparse
import sys

import cv2


def on_trackbar(_value: int) -> None:
    """트랙바 콜백 함수: 값은 메인 루프에서 직접 읽어 사용합니다."""
    return


def main() -> None:
    """트랙바로 임계값을 조절하며 이진화 결과를 실시간 확인하는 예제입니다."""
    parser = argparse.ArgumentParser(description="트랙바 기반 이진화 데모")
    parser.add_argument("--input", default="lenna.png", help="입력 이미지 파일명 (data 폴더 기준)")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="GUI 루프 없이 임계값 128 결과를 1회 계산해 콘솔로만 검증",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    input_path = base_dir / "data" / args.input
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # 1) 이미지를 그레이스케일로 읽습니다.
    gray = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"[ERROR] 이미지를 읽을 수 없습니다: {input_path}")
        sys.exit(1)

    if args.self_check:
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        white_pixels = int((binary == 255).sum())
        print(f"[SELF-CHECK] binary shape={binary.shape}, white_pixels={white_pixels}")
        cv2.imwrite(str(results_dir / "ex4_binary_thresh128.png"), binary)
        print(f"[INFO] 결과 저장 완료: {results_dir}")
        return

    window_name = "Threshold Demo"

    # 2) 데모 윈도우를 만들고, 0~255 범위/초기값 128인 트랙바를 생성합니다.
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Threshold", window_name, 128, 255, on_trackbar)

    while True:
        # 3) 트랙바 현재 값으로 임계값을 적용해 결과 영상을 실시간으로 갱신합니다.
        threshold_value = cv2.getTrackbarPos("Threshold", window_name)
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        cv2.imshow(window_name, binary)

        # 4) 'q' 키를 누르면 루프를 빠져나와 종료합니다.
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break

    # 5) 마지막 이진화 결과를 results 폴더에 저장합니다.
    cv2.imwrite(str(results_dir / "ex4_binary_final.png"), binary)
    print(f"[INFO] 결과 저장 완료: {results_dir}")

    # 6) 열린 모든 창을 닫아 자원을 정리합니다.
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


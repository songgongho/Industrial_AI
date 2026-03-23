from pathlib import Path
import argparse
import sys

import cv2


def main() -> None:
    """Lenna 이미지를 그레이스케일로 변환하고 저장하는 예제입니다."""
    parser = argparse.ArgumentParser(description="Lenna BGR -> Gray 변환 및 저장")
    parser.add_argument("--input", default="Lenna.png", help="입력 이미지 파일명")
    parser.add_argument("--output", default="Lenna_gray.png", help="저장할 그레이스케일 파일명")
    parser.add_argument("--no-gui", action="store_true", help="창 표시 없이 저장/검증만 수행")
    args = parser.parse_args()

    # 현재 스크립트 위치를 기준으로 입력/출력 경로를 고정하면 실행 위치가 달라도 안정적입니다.
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / args.input
    output_path = base_dir / args.output

    # 1) 컬러(BGR)로 이미지를 읽습니다.
    color_img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if color_img is None:
        print(f"[ERROR] 이미지를 읽을 수 없습니다: {input_path}")
        sys.exit(1)

    # 2) BGR 이미지를 그레이스케일로 변환합니다.
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    # 4) 변환한 그레이스케일 이미지를 파일로 저장합니다.
    saved = cv2.imwrite(str(output_path), gray_img)
    if not saved:
        print(f"[ERROR] 그레이스케일 이미지 저장에 실패했습니다: {output_path}")
        sys.exit(1)

    print(f"[INFO] 저장 완료: {output_path}")

    if args.no_gui:
        return

    # 3) 원본 컬러/그레이스케일 이미지를 각각 다른 창에 표시합니다.
    cv2.imshow("Lenna Color", color_img)
    cv2.imshow("Lenna Gray", gray_img)

    # 5) 키 입력 대기 후 모든 창을 닫아 자원을 정리합니다.
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


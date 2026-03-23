from pathlib import Path
import argparse
import sys

import cv2
import numpy as np


def main() -> None:
    """Lenna 이미지를 YUV로 변환하고 채널을 시각화하는 예제입니다."""
    parser = argparse.ArgumentParser(description="Lenna BGR -> YUV 변환 및 채널 시각화")
    parser.add_argument("--input", default="Lenna.png", help="입력 이미지 파일명")
    parser.add_argument("--no-gui", action="store_true", help="창 표시 없이 통계 출력만 수행")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / args.input

    # 1) 이미지를 읽고 실패 시 즉시 종료합니다.
    bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"[ERROR] 이미지를 읽을 수 없습니다: {input_path}")
        sys.exit(1)

    # 2) BGR 이미지를 YUV 색 공간으로 변환합니다.
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

    # 3) YUV 이미지를 Y, U, V 채널로 분리합니다.
    y_channel, u_channel, v_channel = cv2.split(yuv)

    # 5) Y(휘도) 채널의 통계 정보를 계산해 콘솔에 출력합니다.
    y_min = int(np.min(y_channel))
    y_max = int(np.max(y_channel))
    y_mean = float(np.mean(y_channel))
    print(f"[Y 통계] min={y_min}, max={y_max}, mean={y_mean:.2f}")

    if args.no_gui:
        return

    # 4) 각 채널을 별도 창에 동시에 표시합니다.
    cv2.imshow("Y Channel", y_channel)
    cv2.imshow("U Channel", u_channel)
    cv2.imshow("V Channel", v_channel)

    # 6) 키 입력을 기다린 뒤 모든 창을 닫습니다.
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


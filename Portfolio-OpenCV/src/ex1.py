from pathlib import Path
import argparse
import sys

import cv2


def print_top_left_5x5(name: str, channel):
    # 채널 값 확인을 위해 좌상단 5x5 영역만 출력합니다.
    print(f"\n[{name}] top-left 5x5:")
    print(channel[:5, :5])


def main() -> None:
    parser = argparse.ArgumentParser(description="Lenna 이미지를 읽어 BGR/HSV 채널을 확인합니다.")
    parser.add_argument("--no-gui", action="store_true", help="창 표시 없이 콘솔 출력만 수행합니다.")
    args = parser.parse_args()

    # 실행 파일 기준 경로에서 Lenna.png를 찾습니다.
    image_path = Path(__file__).resolve().parent / "Lenna.png"
    image = cv2.imread(str(image_path))

    # 이미지 로딩 실패 시 메시지를 출력하고 프로그램을 종료합니다.
    if image is None:
        print(f"[ERROR] 이미지를 읽을 수 없습니다: {image_path}")
        sys.exit(1)

    # BGR 채널 분리
    blue, green, red = cv2.split(image)
    print_top_left_5x5("Blue Channel", blue)
    print_top_left_5x5("Green Channel", green)
    print_top_left_5x5("Red Channel", red)

    # BGR -> HSV 변환 후 HSV 채널 분리
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv_image)
    print_top_left_5x5("Hue Channel", hue)
    print_top_left_5x5("Saturation Channel", saturation)
    print_top_left_5x5("Value Channel", value)

    if args.no_gui:
        return

    # 원본, BGR 채널, HSV 이미지, HSV 채널을 각각 창으로 띄웁니다.
    cv2.imshow("Original - Lenna", image)
    cv2.imshow("Blue Channel", blue)
    cv2.imshow("Green Channel", green)
    cv2.imshow("Red Channel", red)
    cv2.imshow("HSV Image", hsv_image)
    cv2.imshow("Hue Channel", hue)
    cv2.imshow("Saturation Channel", saturation)
    cv2.imshow("Value Channel", value)

    # 키 입력을 기다린 뒤 열린 모든 창을 닫습니다.
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


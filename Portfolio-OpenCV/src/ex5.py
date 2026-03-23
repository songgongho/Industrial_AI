from pathlib import Path
import argparse
import sys
from typing import Tuple

import cv2


def fail_and_exit(message: str, code: int = 1) -> None:
    """오류 메시지를 출력하고 프로그램을 종료합니다."""
    print(f"[ERROR] {message}")
    sys.exit(code)


def read_image(path: Path):
    """이미지를 BGR 형식으로 읽어 반환합니다. 실패 시 예외를 발생시킵니다."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return img


def show_image(title: str, img) -> None:
    """지정한 제목으로 이미지를 화면에 표시합니다."""
    cv2.imshow(title, img)


def convert_to_gray(img):
    """BGR 이미지를 그레이스케일로 변환합니다."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def convert_to_hsv(img):
    """BGR 이미지를 HSV 색 공간으로 변환합니다."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def convert_to_yuv(img):
    """BGR 이미지를 YUV 색 공간으로 변환합니다."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)


def split_channels(img) -> Tuple:
    """입력 이미지를 채널별로 분리해 튜플로 반환합니다."""
    return cv2.split(img)


def safe_close_windows() -> None:
    """열린 OpenCV 창을 안전하게 닫습니다."""
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        # 창이 없는 환경에서도 예외로 프로그램이 중단되지 않도록 보호합니다.
        pass


def main() -> None:
    """이미지 로드 -> 색 공간 변환 -> 채널 분리/표시 흐름을 제어합니다."""
    parser = argparse.ArgumentParser(description="OpenCV 리팩터링 예제: 함수화 + main 제어")
    parser.add_argument("--input", default="lenna.png", help="입력 이미지 파일명 (data 폴더 기준)")
    parser.add_argument("--no-gui", action="store_true", help="창 표시 없이 콘솔 출력만 수행")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    image_path = base_dir / "data" / args.input
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    try:
        # 1) 원본 이미지 로드
        bgr = read_image(image_path)

        # 2) 색 공간 변환
        gray = convert_to_gray(bgr)
        hsv = convert_to_hsv(bgr)
        yuv = convert_to_yuv(bgr)

        # 3) 채널 분리
        b, g, r = split_channels(bgr)
        h, s, v = split_channels(hsv)
        y, u, v_yuv = split_channels(yuv)

        # no-gui 모드에서는 핵심 정보만 출력해 빠르게 검증합니다.
        if args.no_gui:
            print(f"[INFO] image={image_path}")
            print(f"[INFO] BGR shape={bgr.shape}, GRAY shape={gray.shape}")
            print(f"[INFO] HSV shape={hsv.shape}, YUV shape={yuv.shape}")
            print(f"[INFO] B[0,0]={int(b[0,0])}, H[0,0]={int(h[0,0])}, Y[0,0]={int(y[0,0])}")

        # 변환 결과를 results 폴더에 저장합니다.
        cv2.imwrite(str(results_dir / "ex5_gray.png"), gray)
        cv2.imwrite(str(results_dir / "ex5_hsv.png"), hsv)
        cv2.imwrite(str(results_dir / "ex5_yuv.png"), yuv)
        cv2.imwrite(str(results_dir / "ex5_b_channel.png"), b)
        cv2.imwrite(str(results_dir / "ex5_g_channel.png"), g)
        cv2.imwrite(str(results_dir / "ex5_r_channel.png"), r)
        cv2.imwrite(str(results_dir / "ex5_h_channel.png"), h)
        cv2.imwrite(str(results_dir / "ex5_s_channel.png"), s)
        cv2.imwrite(str(results_dir / "ex5_v_channel.png"), v)
        cv2.imwrite(str(results_dir / "ex5_y_channel.png"), y)
        cv2.imwrite(str(results_dir / "ex5_u_channel.png"), u)
        cv2.imwrite(str(results_dir / "ex5_v_yuv_channel.png"), v_yuv)
        print(f"[INFO] 결과 저장 완료: {results_dir}")

        if args.no_gui:
            return

        # 4) 변환 결과와 채널을 창으로 표시
        show_image("BGR Image", bgr)
        show_image("GRAY Image", gray)
        show_image("HSV Image", hsv)
        show_image("YUV Image", yuv)

        show_image("B Channel", b)
        show_image("G Channel", g)
        show_image("R Channel", r)

        show_image("H Channel", h)
        show_image("S Channel", s)
        show_image("V Channel", v)

        show_image("Y Channel", y)
        show_image("U Channel", u)
        show_image("V(YUV) Channel", v_yuv)

        cv2.waitKey(0)
    except Exception as exc:
        fail_and_exit(str(exc))
    finally:
        # 5) 창 정리를 공통 함수로 처리해 재사용성을 높입니다.
        safe_close_windows()


if __name__ == "__main__":
    main()


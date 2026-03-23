🛠 주요 실습 소스 상세 설명 (Detailed Implementation)
각 실습 파일은 OpenCV의 핵심 기능을 단계별로 구현하고 있으며, argparse를 통해 GUI 모드와 콘솔 모드를 선택적으로 실행할 수 있도록 설계되었습니다.

📂 Ex 01. BGR & HSV 채널 분리 및 분석
"디지털 영상의 데이터 구조를 수치적으로 이해하는 기초 단계"

Key Functions: cv2.split(), cv2.cvtColor()

Description:

Lenna.png를 로드하여 기본 색상 체계인 BGR 채널을 각각 분리

조명 변화에 강인한 HSV(Hue, Saturation, Value) 색 공간으로 변환 후 채널별 특성 파악

각 채널의 좌상단 5x5 행렬 값을 출력하여 픽셀의 정수 데이터 구조를 직접 확인

📂 Ex 02. 그레이스케일 변환 및 데이터 저장
"데이터 경량화 및 전처리를 위한 필수 변환 공정"

Key Functions: cv2.cvtColor(COLOR_BGR2GRAY), cv2.imwrite()

Description:

3채널 컬러 영상을 1채널 Grayscale 영상으로 변환하여 연산 효율성 증대

변환된 영상을 Lenna_gray.png로 로컬에 저장하는 파일 입출력 프로세스 구축

원본 대비 데이터 크기 변화 및 채널 삭제에 따른 시각적 변화 분석

📂 Ex 03. YUV 색 공간 변환 및 휘도 통계 분석
"영상 전송 및 압축 표준인 YUV 체계의 이해"

Key Functions: cv2.cvtColor(COLOR_BGR2YUV), np.mean(), np.std()

Description:

밝기 정보(Y)와 색차 정보(U, V)가 분리된 YUV 색 공간 적용

Y(휘도) 채널의 평균값과 표준편차를 계산하여 영상의 전체적인 밝기 분포를 수치화

인간의 시각이 밝기에 더 민감하다는 특성을 데이터 통계로 검증

📂 Ex 04. 트랙바 기반 실시간 이진화(Thresholding)
"동적 파라미터 튜닝을 통한 최적의 임계값 도출"

Key Functions: cv2.createTrackbar(), cv2.threshold()

Description:

사용자가 슬라이더(Trackbar)를 조절함에 따라 영상이 실시간으로 이진화되는 Interactive GUI 구현

영상 내 배경과 객체를 분리하기 위한 최적의 Threshold 값을 직관적으로 탐색

--self-check 모드를 통해 특정 임계값에서의 흰색 픽셀 개수를 자동 카운팅

📂 Ex 05. 통합 영상 처리 프레임워크 구축
"실습 내용의 모듈화 및 예외 처리를 포함한 통합 솔루션"

Key Functions: try-except 구조, 다중 윈도우 시각화, 모듈화 함수

Description:

앞선 실습(GRAY, HSV, YUV 변환 및 채널 분리)을 하나의 소스 코드로 통합

파일 로드 실패 시의 예외 처리(Exception Handling) 로직을 추가하여 프로그램 안정성 확보

--no-gui 모드를 지원하여 CLI 환경에서도 이미지의 Shape와 픽셀 값을 빠르게 검증할 수 있는 유연한 구조 설계

💡 실행 방법 (How to Run)
각 스크립트는 터미널에서 다음과 같이 옵션을 주어 실행할 수 있습니다.

Bash
# 기본 실행 (GUI 모드)
python ex5.py

# 콘솔 출력 전용 모드
python ex5.py --no-gui

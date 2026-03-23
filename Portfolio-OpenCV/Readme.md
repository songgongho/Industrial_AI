
# [Project] OpenCV 기반 영상 처리 심화 실습

[cite_start]본 프로젝트는 충북대학교 산업인공지능연구센터(INDAI)에서 주관하는 **지능화 캡스톤 프로젝트**의 3주차 과정으로, 컴퓨터 비전의 기초 개념 이해와 OpenCV 라이브러리를 활용한 실전 영상 처리 기술 습득을 목적으로 함 [cite: 1, 127]

## 1. 개요
* **작성자**: 송공호
* [cite_start]**주제**: OpenCV 기반 영상 처리 심화 및 실습 [cite: 1]
* [cite_start]**일시**: 프로젝트 3주차 과정 [cite: 1]

## 2. 주요 학습 및 구현 내용

### [cite_start]2.1 컴퓨터 비전 기초 이해 [cite: 1, 2]
* [cite_start]컴퓨터 비전 정의: 정지 영상 또는 동영상으로부터 정보를 추출하여 사람이 눈으로 보고 인지하는 작업을 컴퓨터가 수행하게 만드는 기술 [cite: 3]
* [cite_start]시스템 구조: 영상 획득(Sensing, 카메라) 및 인식(Recognition, 알고리즘) 과정의 메커니즘 학습 [cite: 3]
* [cite_start]응용 사례 분석: 농업(수확 드론), 의료(혈관 분할), 자율주행, 스마트팩토리(불량 검사), 보안(얼굴 인식) 등 다양한 산업 분야 적용 사례 조사 [cite: 3, 4]

### [cite_start]2.2 OpenCV 환경 구축 및 기본 입출력 [cite: 2]
* [cite_start]개발 환경: Python 기반 OpenCV 라이브러리 설치 및 운용 [cite: 2]
* [cite_start]이미지 입출력: 정지 영상 파일 읽기 및 쓰기 프로세스 구현 [cite: 2]
* [cite_start]영상 스트리밍: 동영상 파일 및 카메라(Webcam) 실시간 영상 입출력 제어 [cite: 2]

### [cite_start]2.3 영상 처리 심화 기술 실습 [cite: 2]
* [cite_start]**영상 색 공간(Color Space)**: RGB, HSV, Gray-scale 등 다양한 색 공간 변환 기법 습득 [cite: 2]
* [cite_start]**색상 영역 검출**: 특정 색상 범위를 지정하여 영상 내 객체를 분리하는 기법 구현 [cite: 2]
* [cite_start]**히스토그램(Histogram)**: 영상의 밝기 분포 분석 및 대비 개선 기술 이해 [cite: 2]
* [cite_start]**영상 필터링(Filtering)**: 커널(Kernel)을 활용한 노이즈 제거 및 특징 추출 필터 적용 [cite: 2]

### [cite_start]2.4 최신 비전 기술 체험 및 분석 [cite: 5]
* [cite_start]**Face Landmarker**: MediaPipe를 활용한 실시간 얼굴 랜드마크 검출 [cite: 5]
* [cite_start]**Image Captioning**: Azure AI Vision을 활용한 이미지 설명 생성 기술 확인 [cite: 8]
* [cite_start]**Visual Question Answering**: 영상 기반 질의응답 시스템(VQA) 작동 원리 이해 [cite: 9]

## 3. 기술 스택
* **Language**: Python
* **Library**: OpenCV (Open Source Computer Vision Library)
* [cite_start]**Framework**: MediaPipe, Azure AI Vision, Teachable Machine [cite: 5, 8, 13]

## 4. 참고 자료
* [cite_start]충북대학교 지능화 캡스톤 프로젝트 3주차 강의안 [cite: 1]
* [cite_start]OpenCV 공식 Tutorials (docs.opencv.org) [cite: 74]
* [cite_start]LearnOpenCV 및 관련 기술 블로그 [cite: 74]

---

### 💡 활용 안내
* 이 내용은 `README.md` 파일로 직접 복사하여 사용하실 수 있습니다
* 실습 과정에서 생성된 **결과 이미지(색상 검출 결과, 필터링 전후 비교 등)**를 각 섹션 하단에 첨부하면 더욱 완성도 높은 포트폴리오가 됩니다
* 문장의 끝에 온점을 생략하고 개조식으로 작성하여 사업기획실의 보고서 형식을 준수하였습니다

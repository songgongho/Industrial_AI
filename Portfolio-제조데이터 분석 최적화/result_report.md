# 제조데이터 분석과 최적화(4주차) 결과보고서

## 1. 과제 개요
- 과목: 제조데이터 분석과 최적화 (4주차: 제조데이터 수집 및 처리 방법)
- 목표: OPC UA 기반으로 제조 센서 데이터를 수집하고, 정제/라벨링/품질검증까지 재현 가능한 실습 환경을 구축
- 범위: 개발환경 구성, 데이터 수집 파이프라인 실행, 결과 파일 생성 및 검증

## 2. 실습 환경 구성
### 2.1 프로젝트 경로
- `E:\2026-1학기\4th_practice`

### 2.2 의존성 명세
- 파일: `E:\2026-1학기\4th_practice\requirements.txt`
- 주요 패키지
  - `asyncua==1.1.5`
  - `numpy==1.26.4`
  - `pandas==2.2.2`
  - `scikit-learn==1.5.2`
  - `matplotlib==3.9.2`
  - `opencv-contrib-python==4.10.0.84`

### 2.3 실행 보조 스크립트
- 환경 구성: `E:\2026-1학기\4th_practice\scripts\setup_env.ps1`
- 환경 점검: `E:\2026-1학기\4th_practice\verify_environment.py`
- 통합 실행기: `E:\2026-1학기\4th_practice\main.py`

## 3. 실습 시나리오 설계
### 3.1 기본 OPC UA 실습
- 서버: `opcua_basic/opc_server.py`
- 클라이언트: `opcua_basic/opc_client.py`
- 내용: 단일 센서(Temperature) 실시간 송수신 및 구독

### 3.2 제조 설비 시뮬레이션 실습
- 서버: `opcua_basic/opc_server_mfg.py`
- 클라이언트: `opcua_basic/opc_client_mfg.py`
- 내용: `Machine_A`의 Temperature/Pressure 데이터 수집 후 CSV 저장

### 3.3 데이터 파이프라인 실습
- 파이프라인: `data_pipeline/data_pipeline.py`
- 검증 포함 파이프라인: `data_pipeline/data_pipeline_check.py`
- 내용: 수집 -> 정제(결측/중복 처리) -> 라벨링 -> 품질검증

### 3.4 심화 정보모델 실습
- 서버: `information_model/advanced_server.py`
- 클라이언트: `information_model/advanced_client.py`
- 내용: 이벤트(AC), 이력(HA), 원격제어(Prog) 실습

## 4. 환경 구성 및 검증 결과
### 4.1 환경 구성 결과
- `setup_env.ps1` 실행으로 프로젝트 로컬 가상환경(`.venv`) 생성 완료
- `requirements.txt` 기반 패키지 설치 완료

### 4.2 모듈 import/버전 점검 결과
- 점검 명령: `python verify_environment.py`
- 결과 요약
  - `asyncua`, `numpy`, `pandas`, `sklearn`, `matplotlib`, `cv2` import 성공
  - OpenCV contrib 기능 확인
    - `SIFT_create`: 지원
    - `xfeatures2d`: 지원

## 5. 데이터 처리 흐름(4주차 관점)
- 1단계 수집(Acquisition): OPC UA 노드에서 온도/압력 데이터 주기 수집
- 2단계 정제(Cleansing): 결측치 제거/처리, 중복 제거, 수치 포맷 정리
- 3단계 라벨링(Labeling): 온도/압력 임계치 기준으로 정상/이상 상태 부여
- 4단계 품질검증(Quality Verification):
  - 유일성: 중복 비율 점검
  - 완전성: 결측 비율 점검
  - 유효성: 허용 범위 이탈 여부 점검
  - 일관성: 라벨 규칙 위반 여부 점검
  - 정확성: 데이터 타입 및 포맷 적합성 점검

## 6. 산출물
- 실행/구성 산출물
  - `requirements.txt`
  - `scripts/setup_env.ps1`
  - `verify_environment.py`
  - `main.py`
  - `README.md`
- 데이터 산출물(실습 실행 시 생성)
  - `manufacturing_sensor_data.csv`
  - `lifecycle_optimized_dataset.csv`
  - `raw_sensor_dataset.csv`
  - `verified_sensor_dataset.csv`

## 7. 결론
- 4주차 실습을 위한 개발환경이 재현 가능하게 구성됨
- OPC UA 데이터 수집부터 정제/라벨링/품질검증까지 단일 프로젝트 내에서 실행 가능
- `main.py` 기반으로 실습 시나리오를 단계별로 반복 수행할 수 있어 보고서/PPT 작성 및 데모에 적합

## 8. 한계 및 개선 계획
- 현재 라벨링 규칙은 임계치 기반(규칙기반)으로 단순화되어 있음
- 향후 개선
  1. 데이터 수집 시간 확대(표본 수 증가)
  2. 품질지표 임계치 튜닝 및 자동 리포트 생성
  3. 모델링 단계(이상탐지/분류) 연계 및 성능 비교

## 9. 재현 절차(부록)
```powershell
Set-Location "E:\2026-1학기\4th_practice"
.\scripts\setup_env.ps1
.\.venv\Scripts\Activate.ps1
python main.py --list
python main.py verify-env
```

제조 설비 수집 실습 예시(터미널 2개):
```powershell
# 터미널 1
Set-Location "E:\2026-1학기\4th_practice"
.\.venv\Scripts\Activate.ps1
python main.py server-mfg
```

```powershell
# 터미널 2
Set-Location "E:\2026-1학기\4th_practice"
.\.venv\Scripts\Activate.ps1
python main.py client-mfg
python main.py pipeline-check
```


# 제조데이터 분석과 최적화 (4주차) 실습 환경

이 폴더는 `OPC UA 기반 제조 데이터 수집 -> 정제/라벨링 -> 품질 검증` 실습을 바로 수행할 수 있도록 정리되어 있습니다.

## 1) 개발환경 구성 (Windows PowerShell)

프로젝트 루트에서 아래 순서로 실행합니다.

```powershell
Set-Location "E:\2026-1학기\4th_practice"
.\scripts\setup_env.ps1
```

가상환경 수동 활성화가 필요하면:

```powershell
Set-Location "E:\2026-1학기\4th_practice"
.\.venv\Scripts\Activate.ps1
```

환경 점검:

```powershell
python verify_environment.py
```

## 2) 통합 실행기 (`main.py`)

사용 가능한 시나리오 목록:

```powershell
python main.py --list
```

예시 실행:

```powershell
python main.py server-mfg
python main.py client-mfg
python main.py pipeline
python main.py pipeline-check
python main.py adv-server
python main.py adv-client
python main.py verify-env
```

## 3) 실습 권장 순서 (4주차 PDF 흐름 기준)

1. `server-mfg` 실행 (가상 설비/센서 데이터 발생)
2. 다른 터미널에서 `client-mfg` 실행 (수집 확인)
3. `pipeline` 또는 `pipeline-check` 실행 (정제/라벨링/품질 검증)
4. 필요 시 `adv-server` + `adv-client`로 AC/HA/Prog 심화 실습

## 4) 생성 결과물

- `manufacturing_sensor_data.csv`: 기초 수집 결과
- `lifecycle_optimized_dataset.csv`: 정제/라벨링 결과
- `raw_sensor_dataset.csv`: 정제 전 원시 데이터
- `verified_sensor_dataset.csv`: 품질 점검 포함 최종 결과

## 5) 주요 의존성

- `asyncua`: OPC UA 통신
- `pandas`, `numpy`: 데이터 처리
- `scikit-learn`: 결측치 대치/전처리 확장 대비
- `matplotlib`: 분석 그래프 작성 대비
- `opencv-contrib-python`: `import cv2` 및 contrib 기능 사용


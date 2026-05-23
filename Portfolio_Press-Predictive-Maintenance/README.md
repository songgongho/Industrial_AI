# Press MDB 불량/이상 분석 및 예지보전 리포트

Press 로그 MDB 파일(`Press Profile log/*.mdb`)을 읽어 공정 이상을 탐지하고,
불량 유형/위험도/예지보전 이벤트를 자동으로 보고서(`outputs/report.md`)로 생성합니다.

## 주요 기능
- MDB 파일 자동 탐색 및 테이블 자동 선택
- 불량 라벨 컬럼 자동 추론(`result`, `defect` 계열)
- 라벨 컬럼이 없을 때 이상점수 기반 `anomaly_fallback` 적용
- 불량 유형 집계, 파레토 차트, 파일/설비별 위험도 분석
- PM(예지보전) 지표 생성: `pm_risk_score`, `pm_alert_flag`, `pm_event_id`, `pm_lead_time_min`
- Markdown 리포트 + CSV/PNG 산출물 자동 저장

## 프로젝트 구조
- `main.py`: CLI 실행 진입점
- `src/press_analysis/io_mdb.py`: MDB 연결/로딩
- `src/press_analysis/features.py`: 컬럼 추론, 이상/PM 피처 생성
- `src/press_analysis/classify.py`: 집계/시각화/리포트 생성
- `tests/test_pipeline.py`: 스모크 테스트

## 빠른 시작 (PowerShell)
```powershell
cd "E:\2026-1학기\PythonProject_Press"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 동작 확인 (스모크 테스트)
```powershell
python tests\test_pipeline.py
```

## 실제 MDB 분석 실행
```powershell
python main.py --input-dir "Press Profile log" --output-dir "outputs"
```

## 주요 산출물
### 리포트
- `outputs/report.md`

### 테이블(CSV)
- `outputs/tables/summary.csv`
- `outputs/tables/defect_type_counts.csv`
- `outputs/tables/defect_type_pareto.csv`
- `outputs/tables/defect_rate_by_machine.csv`
- `outputs/tables/defect_rate_by_source_file.csv`
- `outputs/tables/primary_anomaly_feature_contrib.csv`
- `outputs/tables/pm_event_timeline.csv`
- `outputs/tables/pm_machine_risk_summary.csv`
- `outputs/tables/pm_alert_leadtime_summary.csv`

### 그래프(PNG)
- `outputs/figures/top_defect_types.png`
- `outputs/figures/defect_type_pareto.png`
- `outputs/figures/pm_risk_timeline.png`

## 리포트 해석 가이드
- `label_strategy`가 `explicit_labels`이면 DB 내 불량 라벨을 직접 사용한 결과입니다.
- `label_strategy`가 `anomaly_fallback`이면 센서 이상점수 기반 추정 결과입니다.
- `primary_anomaly_feature_contrib.csv` 상위 센서를 우선 점검 대상으로 삼는 것을 권장합니다.
- `pm_event_timeline.csv`의 이벤트 시작 시각을 기준으로 사전 정비 정책을 설정할 수 있습니다.

## 실행 조건 / 주의사항
- Windows Access ODBC 드라이버 필요(일반적으로 `Microsoft Access Driver (*.mdb, *.accdb)`).
- Python 비트수와 ODBC 드라이버 비트수(x64/x86)가 일치해야 합니다.
- 로그 콘솔의 `meta` 출력에서 추론 컬럼(`time_col`, `machine_col` 등)을 먼저 확인하세요.


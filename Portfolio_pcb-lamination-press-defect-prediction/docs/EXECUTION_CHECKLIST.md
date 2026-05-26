# PCB Press 공정 분석 - 실행 액션 플랜 및 체크리스트

**목적**: 고객사 데이터 수신 후 즉시 실행할 수 있는 단계별 지침  
**작성일**: 2026년 5월 26일  
**대상**: 프로젝트팀 (데이터 분석가, ML 엔지니어, 연구원)

---

## PHASE 1: 데이터 수신 & 기초 검증 (Week 1-2)

### Task 1.1: 데이터 수신 & 저장 구조화

#### 목표
고객사로부터 받은 데이터를 일관된 폴더 구조로 정리

#### 실행 단계

```bash
# 1. 폴더 구조 생성
mkdir -p data/customer/{raw,processed,validated,archived}
mkdir -p data/customer/logs

# 2. 데이터 파일 분류 및 저장
# (고객사에서 받은 파일들을 아래대로 정렬)
data/customer/raw/
├── quality_daily.csv         # 일별 품질 현황
├── equipment_state.csv       # 설비 가동/비가동
├── press_alarms.csv          # PRESS 알람 이력
├── recipe_changes.csv        # 설정값 변화
├── lot_panel_mapping.csv     # LOT/PANEL/CYCLE 매핑
├── maintenance_logs.csv      # 유지보수 이력
├── operator_info.csv         # 작업자 정보
└── sensor_timeseries.parquet # 센서 시계열 (대용량 시 Parquet 추천)

# 3. 메타데이터 기록
touch data/customer/METADATA.json
cat > data/customer/METADATA.json << 'EOF'
{
  "data_received_date": "2026-05-26",
  "customer": "NeoTech",
  "files": {
    "quality_daily": {"rows": null, "columns": null, "date_range": null},
    ...
  },
  "data_owner_contact": "customer@neotech.com",
  "confidentiality": "Restricted"
}
EOF
```

#### 체크리스트
- [ ] 모든 7가지 데이터셋 파일 확인
- [ ] 파일 포맷 검증 (CSV, Parquet, Excel 등)
- [ ] 파일 크기 기록
- [ ] 메타데이터 JSON 작성
- [ ] Git LFS 또는 S3 등에 백업 (민감 정보 보호)

---

### Task 1.2: 데이터 품질 검증 스크립트 실행

#### 목표
누락값, 범위 이상, 중복값, 시간 일관성 자동 검사

#### 스크립트 생성 및 실행

```python
# scripts/validate_customer_data.py

import pandas as pd
import numpy as np
from datetime import datetime
import json

def validate_customer_data(data_dir='data/customer/raw'):
    """
    고객 데이터 품질 검증 종합 함수
    
    Returns:
        validation_report (dict): 검증 결과 상세 보고서
    """
    
    report = {
        "validation_timestamp": datetime.now().isoformat(),
        "files": {},
        "summary": {}
    }
    
    files_to_validate = [
        ('quality_daily.csv', ['date', 'shift', 'good_qty', 'scrap_qty', 'defect_code']),
        ('equipment_state.csv', ['machine_id', 'timestamp', 'state', 'downtime_min']),
        ('press_alarms.csv', ['alarm_time', 'alarm_code', 'severity', 'duration_sec']),
        ('recipe_changes.csv', ['change_timestamp', 'parameter_name', 'old_value', 'new_value']),
        ('lot_panel_mapping.csv', ['lot_id', 'panel_id', 'cycle_id', 'timestamp']),
        ('maintenance_logs.csv', ['maintenance_date', 'component', 'work_type']),
        ('operator_info.csv', ['shift', 'operator_id', 'experience_level']),
    ]
    
    total_issues = 0
    
    for filename, required_cols in files_to_validate:
        filepath = f"{data_dir}/{filename}"
        file_report = {
            "file": filename,
            "status": "PASS",
            "issues": []
        }
        
        try:
            # 파일 로드
            df = pd.read_csv(filepath)
            
            # 1. 컬럼 검증
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                file_report["issues"].append(f"Missing columns: {missing_cols}")
                file_report["status"] = "FAIL"
            
            # 2. 누락값 검증
            missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
            for col, pct in missing_pct.items():
                if pct > 5:  # 5% 이상 누락이면 경고
                    file_report["issues"].append(
                        f"Column '{col}' has {pct:.1f}% missing values"
                    )
            
            # 3. 중복값 검증
            duplicate_rows = df.duplicated().sum()
            if duplicate_rows > 0:
                file_report["issues"].append(
                    f"Found {duplicate_rows} duplicate rows"
                )
            
            # 4. 시간 일관성 검증 (타임스탐프 있는 파일만)
            timestamp_cols = [c for c in df.columns if 'time' in c.lower()]
            for ts_col in timestamp_cols:
                try:
                    ts_series = pd.to_datetime(df[ts_col])
                    # 역순 확인
                    if not ts_series.is_monotonic_increasing:
                        file_report["issues"].append(
                            f"Column '{ts_col}' is not monotonic increasing"
                        )
                except Exception as e:
                    file_report["issues"].append(
                        f"Column '{ts_col}' datetime parsing failed: {str(e)}"
                    )
            
            # 5. 범위 검증 (양수 컬럼)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if 'qty' in col.lower() or 'time' in col.lower():
                    if (df[col] < 0).any():
                        file_report["issues"].append(
                            f"Column '{col}' contains negative values"
                        )
            
            # 6. 통계 요약
            file_report["summary"] = {
                "rows": len(df),
                "columns": len(df.columns),
                "date_range": None,
                "unique_values": {col: df[col].nunique() for col in df.columns[:5]}
            }
            
            # 날짜 범위 추출
            if timestamp_cols:
                ts_col = timestamp_cols[0]
                ts_series = pd.to_datetime(df[ts_col])
                file_report["summary"]["date_range"] = {
                    "start": ts_series.min().isoformat(),
                    "end": ts_series.max().isoformat(),
                    "days": (ts_series.max() - ts_series.min()).days
                }
            
            if file_report["status"] == "FAIL":
                total_issues += len(file_report["issues"])
            
        except FileNotFoundError:
            file_report["status"] = "FILE_NOT_FOUND"
            file_report["issues"].append(f"File not found: {filepath}")
            total_issues += 1
        except Exception as e:
            file_report["status"] = "ERROR"
            file_report["issues"].append(f"Error processing file: {str(e)}")
            total_issues += 1
        
        report["files"][filename] = file_report
    
    # 종합 요약
    report["summary"] = {
        "total_files": len(files_to_validate),
        "files_passed": sum(1 for f in report["files"].values() if f["status"] == "PASS"),
        "files_failed": sum(1 for f in report["files"].values() if f["status"] != "PASS"),
        "total_issues": total_issues,
        "recommendation": "Data ready for analysis" if total_issues == 0 else "Data needs cleaning before analysis"
    }
    
    # 보고서 저장
    report_path = f"{data_dir}/../validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Validation Report saved to: {report_path}")
    print(f"   Issues found: {total_issues}")
    print(f"   Status: {report['summary']['recommendation']}")
    
    return report

if __name__ == "__main__":
    report = validate_customer_data()
    
    # 보고서 출력
    print("\n" + "="*60)
    print("CUSTOMER DATA VALIDATION REPORT")
    print("="*60)
    for filename, file_report in report["files"].items():
        status_icon = "✅" if file_report["status"] == "PASS" else "⚠️"
        print(f"\n{status_icon} {filename}")
        print(f"   Status: {file_report['status']}")
        if file_report["issues"]:
            for issue in file_report["issues"][:3]:  # 처음 3개만 출력
                print(f"   - {issue}")
        if file_report["summary"]:
            print(f"   Rows: {file_report['summary'].get('rows', 'N/A')}")
```

#### 실행

```bash
python scripts/validate_customer_data.py
```

#### 산출물
- `data/customer/validation_report.json` - 상세 검증 결과
- 콘솔 출력 - 빠른 요약

#### 체크리스트
- [ ] 스크립트 실행 완료
- [ ] validation_report.json 생성 확인
- [ ] 각 파일의 상태 확인 (PASS/FAIL)
- [ ] 문제점 목록 작성 및 고객사 재요청 검토

---

### Task 1.3: 데이터 동기화 및 시간 정렬

#### 목표
여러 데이터셋의 타임스탬프를 맞춰서 시계열 분석 준비

#### 스크립트

```python
# scripts/synchronize_customer_data.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def synchronize_timeseries(data_dir='data/customer/raw', output_dir='data/customer/processed'):
    """
    고객 데이터의 타임스탬프를 통일하고 시간 정렬
    
    전략:
    1. 각 테이블의 타임스탬프 컬럼 정규화 (UTC, 분 단위)
    2. LOT/PANEL/CYCLE 경계 표시
    3. 센서 데이터 리샘플링 (일관된 빈도)
    4. 통합 마스터 인덱스 생성
    """
    
    print("🔄 Synchronizing customer timeseries data...")
    
    # 1. 센서 시계열 데이터 로드 및 리샘플링
    print("  [1/5] Loading sensor data...")
    sensor_df = pd.read_parquet(f'{data_dir}/sensor_timeseries.parquet')
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    
    # 1분 단위로 리샘플링 (결측값은 선형 보간)
    sensor_df_resampled = sensor_df.set_index('timestamp').resample('1min').interpolate(method='linear')
    sensor_df_resampled = sensor_df_resampled.reset_index()
    
    # 2. 품질 데이터 (일별 → 분 단위로 확장)
    print("  [2/5] Expanding quality data...")
    quality_df = pd.read_csv(f'{data_dir}/quality_daily.csv')
    quality_df['date'] = pd.to_datetime(quality_df['date'])
    
    # 각 날의 시작 시간으로 조정
    quality_df['timestamp_start'] = quality_df['date'] + \
        pd.to_timedelta(quality_df['shift'].map({
            '1': '06:00', '2': '14:00', '3': '22:00'
        }))
    quality_df['timestamp_end'] = quality_df['timestamp_start'] + timedelta(hours=8)
    
    # 분 단위 타임스탬프 생성
    quality_expanded = []
    for _, row in quality_df.iterrows():
        timestamps = pd.date_range(
            start=row['timestamp_start'],
            end=row['timestamp_end'],
            freq='1min'
        )
        for ts in timestamps:
            quality_expanded.append({
                'timestamp': ts,
                'quality_label': row.get('defect_code', 'UNKNOWN'),
                'shift': row['shift'],
                'good_qty': row['good_qty'],
                'scrap_qty': row['scrap_qty']
            })
    quality_expanded_df = pd.DataFrame(quality_expanded)
    
    # 3. 설비 상태 데이터 병합
    print("  [3/5] Merging equipment state...")
    equipment_df = pd.read_csv(f'{data_dir}/equipment_state.csv')
    equipment_df['timestamp'] = pd.to_datetime(equipment_df['timestamp'])
    
    # 3분 단위로 리샘플링 (덜 빈번한 데이터)
    equipment_resampled = equipment_df.set_index('timestamp').resample('3min').ffill()
    equipment_resampled = equipment_resampled.reset_index()
    
    # 4. 알람 데이터 이벤트 추가
    print("  [4/5] Adding alarm events...")
    alarms_df = pd.read_csv(f'{data_dir}/press_alarms.csv')
    alarms_df['alarm_time'] = pd.to_datetime(alarms_df['alarm_time'])
    alarms_df = alarms_df.rename(columns={'alarm_time': 'timestamp'})
    alarms_df['event_type'] = 'ALARM'
    
    # 5. 마스터 타임스탬프 생성 및 병합
    print("  [5/5] Creating master index...")
    
    min_time = min(
        sensor_df_resampled['timestamp'].min(),
        quality_expanded_df['timestamp'].min(),
        equipment_resampled['timestamp'].min()
    )
    max_time = max(
        sensor_df_resampled['timestamp'].max(),
        quality_expanded_df['timestamp'].max(),
        equipment_resampled['timestamp'].max()
    )
    
    master_index = pd.date_range(start=min_time, end=max_time, freq='1min')
    master_df = pd.DataFrame({'timestamp': master_index})
    
    # Left join으로 통합 (센서 데이터 중심)
    master_df = master_df.merge(
        sensor_df_resampled, on='timestamp', how='left'
    )
    master_df = master_df.merge(
        quality_expanded_df, on='timestamp', how='left'
    )
    master_df = master_df.merge(
        equipment_resampled, on='timestamp', how='left'
    )
    
    # LOT/PANEL/CYCLE 경계 추가
    mapping_df = pd.read_csv(f'{data_dir}/lot_panel_mapping.csv')
    mapping_df['timestamp'] = pd.to_datetime(mapping_df['timestamp'])
    mapping_df = mapping_df[['timestamp', 'lot_id', 'panel_id', 'cycle_id']].drop_duplicates()
    
    # 가장 가까운 timestamp로 forward fill
    master_df = master_df.merge(
        mapping_df, on='timestamp', how='left'
    ).fillna(method='ffill')
    
    # 저장
    master_df.to_parquet(f'{output_dir}/master_synchronized.parquet', index=False)
    
    print(f"\n✅ Synchronized data saved to: {output_dir}/master_synchronized.parquet")
    print(f"   Time range: {min_time} to {max_time}")
    print(f"   Total records: {len(master_df):,}")
    print(f"   Missing rate: {master_df.isnull().sum().sum() / (len(master_df) * len(master_df.columns)) * 100:.1f}%")
    
    return master_df

if __name__ == "__main__":
    master_df = synchronize_timeseries()
    print("\n" + master_df.head(10).to_string())
```

#### 실행

```bash
python scripts/synchronize_customer_data.py
```

#### 체크리스트
- [ ] 센서 데이터 1분 단위 리샘플링 완료
- [ ] 품질 데이터 시간 확장 완료
- [ ] master_synchronized.parquet 생성 확인
- [ ] 마스터 인덱스의 시간 연속성 확인

---

## PHASE 2: 탐색적 데이터 분석 (EDA) (Week 3-4)

### Task 2.1: 기초 통계 분석

#### 목표
센서, 불량, 설비 데이터의 기초 통계 및 분포 파악

#### 스크립트

```python
# scripts/eda_customer_data.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

def eda_customer_data(master_df, output_dir='outputs/eda'):
    """
    고객 데이터 탐색적 분석
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("📊 Exploratory Data Analysis (EDA)...")
    
    eda_results = {
        "timestamp": datetime.now().isoformat(),
        "data_shape": {"rows": len(master_df), "columns": len(master_df.columns)},
        "sensor_stats": {},
        "quality_analysis": {},
        "equipment_analysis": {}
    }
    
    # 1. 센서 데이터 기초 통계
    sensor_cols = [c for c in master_df.columns if 'T_' in c or 'P_' in c or 'H_' in c]
    
    for sensor in sensor_cols[:10]:  # 처음 10개만 (너무 많으면 출력 너무 김)
        stats_dict = {
            "mean": float(master_df[sensor].mean()),
            "std": float(master_df[sensor].std()),
            "min": float(master_df[sensor].min()),
            "max": float(master_df[sensor].max()),
            "quantile_25": float(master_df[sensor].quantile(0.25)),
            "quantile_50": float(master_df[sensor].quantile(0.50)),
            "quantile_75": float(master_df[sensor].quantile(0.75))
        }
        eda_results["sensor_stats"][sensor] = stats_dict
        
        # 정규성 검정 (Shapiro-Wilk)
        sample = master_df[sensor].dropna().sample(min(1000, len(master_df[sensor])))
        stat, p_value = stats.shapiro(sample)
        eda_results["sensor_stats"][sensor]["normality_p_value"] = float(p_value)
    
    # 2. 불량 분석
    if 'quality_label' in master_df.columns:
        quality_counts = master_df['quality_label'].value_counts()
        eda_results["quality_analysis"] = {
            "defect_codecounts": quality_counts.to_dict(),
            "defect_rate": float(quality_counts.get('DEFECT', 0) / len(master_df) * 100),
            "top_defects": quality_counts.head(5).to_dict()
        }
    
    # 3. 설비 상태 분석
    if 'state' in master_df.columns:
        state_counts = master_df['state'].value_counts()
        eda_results["equipment_analysis"] = {
            "state_distribution": state_counts.to_dict(),
            "uptime_pct": float(state_counts.get('RUN', 0) / len(master_df) * 100)
        }
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 불량률 시계열
    if 'quality_label' in master_df.columns:
        defect_series = (master_df['quality_label'] != 'GOOD').astype(int)
        defect_rolling = defect_series.rolling(window=1440).mean() * 100  # 1일 주기
        axes[0, 0].plot(defect_rolling.index, defect_rolling.values)
        axes[0, 0].set_title('Defect Rate (Rolling 24hr)')
        axes[0, 0].set_ylabel('Defect Rate (%)')
    
    # 센서 분포
    if sensor_cols:
        master_df[sensor_cols[0]].hist(bins=50, ax=axes[0, 1])
        axes[0, 1].set_title(f'Distribution: {sensor_cols[0]}')
    
    # 상관관계 히트맵
    corr_cols = sensor_cols[:5] + (['quality_label_numeric'] if 'quality_label' in master_df.columns else [])
    if len(corr_cols) > 1:
        corr_matrix = master_df[corr_cols].corr()
        sns.heatmap(corr_matrix, ax=axes[1, 0], cmap='coolwarm')
        axes[1, 0].set_title('Sensor Correlation Matrix')
    
    # 설비 가동률
    if 'state' in master_df.columns:
        state_dist = master_df['state'].value_counts()
        state_dist.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Equipment State Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/eda_overview.png', dpi=150, bbox_inches='tight')
    print(f"✅ EDA plot saved: {output_dir}/eda_overview.png")
    
    # JSON 보고서 저장
    with open(f'{output_dir}/eda_report.json', 'w') as f:
        json.dump(eda_results, f, indent=2)
    
    return eda_results

if __name__ == "__main__":
    master_df = pd.read_parquet('data/customer/processed/master_synchronized.parquet')
    eda_customer_data(master_df)
```

#### 체크리스트
- [ ] EDA 스크립트 실행 완료
- [ ] eda_overview.png 생성 확인
- [ ] eda_report.json 생성 확인
- [ ] 불량률, 센서 범위, 상관관계 파악 완료

---

##

 PHASE 3: 인과 추론 (Causal Discovery) (Week 5-7)

### Task 3.1: PCMCI 알고리즘 적용

#### 목표
센서 간 시간 지연을 포함한 인과관계 방향 규명

#### 스크립트 체크리스트

```python
# ml/run_pcmci_discovery.py

# 1. Tigramite 라이브러리 설치
# pip install tigramite

# 2. 데이터 준비
# master_df를 [시간, 센서] 배열로 변환

# 3. PCMCI 실행
from tigramite.data import Data
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

# 4. 결과 해석 및 시각화
# - 가중치 행렬 W 추출
# - DAG 엣지 도시
# - 시간 지연 정보 기록

# 5. 결과 저장
# causal_graph.json, causal_weights.csv
```

#### 실행 예상 결과
```
Causal Relationships Found:
- Temperature_Probe_1(t-0.5h) → Defect (p=0.001)
- Pressure(t) → Temperature_Probe_2(t+0.2h) (p=0.01)
- Humidity(t-1h) → Quality_Metric (p=0.05)

Top 5 Causes of Defect:
1. Temperature (lag=30min, weight=0.8)
2. Pressure variance (lag=0min, weight=0.6)
3. Humidity change (lag=60min, weight=0.4)
...
```

#### 체크리스트
- [ ] Tigramite 설치 완료
- [ ] PCMCI 역 실행 완료
- [ ] causal_graph.JSON 생성 확인
- [ ] DAG 시각화 이미지 생성

---

## PHASE 4: 선행 연구 병렬 학습

### Learning Checklist 4.1: 인과 추론 (Causal Inference)

자세한 내용은 **[DATA_ANALYSIS_FRAMEWORK.md](#section-3-선행-연구-및-학습-방향)** 의 Section 3.2를 참고

#### L1-1: 인과 그래프 기초 (1주)
```
□ Judea Pearl의 "Book of Why" 읽기 (Ch 1-3)
□ Causal DAG 그리기 연습 (3-노드 예제)
□ d-separation 규칙 이해
□ 우리 PCB Press 데이터의 DAG 초안 그리기
  예: Temperature → Pressure → Humidity → Defect
```

**산출물**: `docs/our_causal_dag_draft.png`

#### L1-2: Confounding & Collider (1주)
```
□ 교란 변수(Confounding) 개념 이해
  예: Quality라는 교란 변수가 Temperature와 Pressure를 동시에 영향
□ Collider 개념
  예: Alarm이 Temperature와 Pressure의 결과이지만...
□ d-separation으로 독립성 판단
□ 실제 센서 데이터에 적용
```

**산출물**: `src/research/confounding_analysis.py`

#### L1-3: PCMCI 알고리즘 (2주)
```
□ PCMCI 논문(*Runge et al., 2019*) 읽기
□ Conditional Mutual Information (CMI) 개념
□ Tigramite 튜토리얼 (예제 데이터로)
□ 우리 고객 데이터에 PCMCI 적용
□ 결과 검증 (Domain expert와 함께)
```

**산출물**: `ml/run_pcmci_discovery.py`, `outputs/pcmci_results/`

---

## FINAL: 프로젝트 커밋 및 문서화

### Task F.1: GitHub 커밋

```bash
# 1. 새 브랜치 생성
git checkout -b feature/customer-data-analysis

# 2. 생성된 파일 Stage
git add docs/DATA_ANALYSIS_FRAMEWORK.md
git add scripts/validate_customer_data.py
git add scripts/synchronize_customer_data.py
git add scripts/eda_customer_data.py
git add ml/run_pcmci_discovery.py

# 3. 커밋 메시지 작성
git commit -m "docs: Add comprehensive data analysis framework and action plans

## Summary
- Created DATA_ANALYSIS_FRAMEWORK.md with 6 sections:
  - 고객사 데이터 요청 7가지 항목 상세 분석
  - 데이터 수신 후 분석 계획 (5가지 인사이트 도출 방법)
  - 선행 연구 및 학습 경로 (5가지 학습 트랙)
  - 주요 알고리즘 설명 (MS-CDPNet, PCMCI, NOTEARS, GNN, SHAP)
  - 현재 진행 현황 (40% 완료)
  - 향후 9개월 로드맵

- Created EXECUTION_CHECKLIST.md with actionable steps:
  - Phase 1: 데이터 수신 & 검증 (2주)
  - Phase 2: EDA (2주)
  - Phase 3: 인과 추론 (3주)
  - Phase 4: 병렬 학습 (지속적)

## Key Deliverables
- Validation script: scripts/validate_customer_data.py
- Synchronization script: scripts/synchronize_customer_data.py
- EDA script: scripts/eda_customer_data.py
- PCMCI runner: ml/run_pcmci_discovery.py

## Next Steps
- 고객사 데이터 수신 후 Task 1.1 실행
- 병렬로 Learning Path 1-5 시작
- 주간 진행도 리뷰 (매주 월요일)

Refs: #21, #25
Co-authored-by: GitHub Copilot"

# 4. 원격 저장소에 푸시
git push origin feature/customer-data-analysis

# 5. Pull Request 생성 (GitHub 웹)
# → feat/customer-data-analysis → main으로 PR 생성
```

### Task F.2: 문서 인덱싱

```markdown
# docs/MASTER_INDEX_DATA_ANALYSIS.md 업데이트

추가 항목:
- [DATA_ANALYSIS_FRAMEWORK.md](./DATA_ANALYSIS_FRAMEWORK.md) - 종합 분석 프레임워크
- [EXECUTION_CHECKLIST.md](./EXECUTION_CHECKLIST.md) - 단계별 실행 지침

## 읽기 순서 추천
1. [README.md](../README.md) - 프로젝트 개요
2. **[DATA_ANALYSIS_FRAMEWORK.md](./DATA_ANALYSIS_FRAMEWORK.md)** ← NEW (20-30분)
   - 대상: 전체 팀
   - 목적: 전체 전략 이해
3. **[EXECUTION_CHECKLIST.md](./EXECUTION_CHECKLIST.md)** ← NEW (15-20분)
   - 대상: 실행 담당자 (데이터 분석가, ML 엔지니어)
   - 목적: 실제 작업 리스트 확인
4. [RESEARCH_ROADMAP_2026.md](./RESEARCH_ROADMAP_2026.md) - 상세 로드맵
5. [literature/](./literature/) - 선행 연구 분석
```

### Task F.3: 최종 커밋

```bash
git add docs/DATA_ANALYSIS_FRAMEWORK.md
git add docs/EXECUTION_CHECKLIST.md
git commit -m "docs: Add execution checklist for customer data analysis"
git push origin feature/customer-data-analysis
```

---

## 최종 요약 체크리스트

### 전체 액션 플랜 요약
```
□ PHASE 1: 데이터 수신 & 검증 (Week 1-2)
  └─ □ Task 1.1: 데이터 저장 구조화
  └─ □ Task 1.2: 품질 검증 스크립트 실행
  └─ □ Task 1.3: 시간 동기화

□ PHASE 2: EDA 분석 (Week 3-4)
  └─ □ Task 2.1: 기초 통계 분석
  └─ □ 불량률, 센서 분포, 상관관계 파악

□ PHASE 3: 인과 추론 (Week 5-7)
  └─ □ Task 3.1: PCMCI 알고리즘 적용
  └─ □ 센서 간 인과관계 DAG 추출

□ PHASE 4: 병렬 학습 (지속적)
  └─ □ L1: 인과 추론 (2-3주)
  └─ □ L2: 설명 가능성 (2-3주)
  └─ □ L3: 시계열 ML (3-4주)
  └─ □ L4: GNN (2-3주)
  └─ □ L5: 최적화 (2-3주)

□ FINAL: GitHub 커밋 및 문서화
  └─ □ 새 문서 추가
 └─ □ 스크립트 커밋
  └─ □ PR 생성 및 병합
```

---

**문서 버전**: 1.0  
**작성일**: 2026년 5월 26일  
**다음 검토 예정**: 2026년 6월 1일 (고객사 데이터 수신 직후)



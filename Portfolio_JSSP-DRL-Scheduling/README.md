# 🏭 JSSP 스케줄링 최적화: ILP vs 휴리스틱 vs 심층강화학습 (JSSP Scheduling Optimization)

Job Shop Scheduling Problem(JSSP)을 ILP(정수 선형 계획법), 휴리스틱 기법, 심층강화학습(DRL/PPO)으로 각각 풀고 Makespan 성능을 비교 분석한 프로젝트입니다. SimPy 기반 공정 시뮬레이션도 병행 구현하였습니다.

---

## 📌 프로젝트 개요

- **분야**: 제조 스케줄링 최적화 / 심층강화학습
- **문제**: JSSP - 여러 작업(Job)을 여러 기계(Machine)에 최적 배정하여 전체 완료 시간(Makespan) 최소화
- **비교 대상**: ILP 최적해 vs 휴리스틱 vs DRL(PPO) vs SimPy 시뮬레이션

---

## 📂 파일 구조

```
Portfolio_JSSP-DRL-Scheduling/
├── JSSP_easy/                     # 소규모 JSSP (비교적 단순한 문제)
│   ├── step1_ilp_optimal.py       # ILP 절대 최적해 (PuLP/MILP)
│   ├── step2_heuristics.py        # 휴리스틱 기법 (SPT, LPT 등)
│   ├── step3_drl_basic.py         # 기본 DRL (PPO) 스케줄러
│   ├── step4_drl_advanced.py      # 개선 DRL 스케줄러
│   ├── utils.py                   # 공통 유틸리티
│   └── main_compare.py            # 통합 비교 실행
├── JSSP_hard/                     # 대규모 JSSP (복잡한 문제)
│   └── (JSSP_easy 동일 구조)
├── simpy/                         # SimPy 기반 공정 시뮬레이션
│   ├── step1_simpy_basics.py      # SimPy 기초 시뮬레이션
│   ├── step2_simpy_gui.py         # GUI 연동 시뮬레이션
│   ├── step3_timetable_gantt.py   # 간트 차트 생성
│   └── step4_bottleneck_analysis.py # 병목 분석
├── resource_vs_scheduling/        # 자원 배분 vs 스케줄링 비교
└── results_report.pdf             # 12주차 최종 결과 보고서
```

---

## 🔑 핵심 기술

- **ILP (MILP)**: PuLP 라이브러리로 수학적 최적해 도출 → 평가 기준(Ground Truth) 역할
- **휴리스틱**: SPT(Shortest Processing Time), LPT 등 규칙 기반 빠른 근사해
- **DRL/PPO**: 강화학습 에이전트가 반복 경험으로 스케줄링 정책 학습
- **SimPy**: 이산 사건 시뮬레이션으로 실제 공정 흐름 및 병목 분석

---

## 📊 결과 요약

- ILP: 소규모 문제에서 증명된 최적해 제공, 대규모에서는 계산 시간 과다
- 휴리스틱: 빠른 근사해, 최적해 대비 10~20% 성능 손실
- DRL(PPO): 학습 후 ILP 근접 성능 달성, 대규모 문제에서 확장성 우수
- SimPy: 실제 생산 라인의 병목 공정 시각화 및 개선 포인트 도출

---

## 📝 사용 기술 스택

- Python, PuLP (MILP), SimPy
- PyTorch (PPO/강화학습)
- Matplotlib, Pandas

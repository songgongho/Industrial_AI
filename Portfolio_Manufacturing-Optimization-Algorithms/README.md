# ⚙️ 제조 최적화 알고리즘 비교 (Manufacturing Optimization Algorithms)

제조 공정의 생산 계획 문제를 LP, IP, DP, Greedy, GA, AI(강화학습) 등 6가지 최적화 알고리즘으로 풀고 성능을 비교 분석한 프로젝트입니다.

---

## 📌 프로젝트 개요

- **분야**: 제조 최적화 / 운영 연구 (OR)
- **문제**: 제조 환경에서의 생산 계획 최적화 (Production Planning)
- **목표**: 동일 문제를 6가지 알고리즘으로 풀어 각 기법의 특성과 적용 한계 비교

---

## 🛠️ 구현 알고리즘

| 번호 | 알고리즘 | 파일 | 특징 |
|------|----------|------|------|
| 1 | LP (선형 계획법) | `1_lp_solver.py` | 연속 변수, 전역 최적, 고속 |
| 2 | IP (정수 계획법) | `2_ip_solver.py` | 정수 변수, 전역 최적 보장 |
| 3 | DP (동적 계획법) | `3_dp_solver.py` | 부분 문제 분할, 최적 부분구조 |
| 4 | Greedy (탐욕 알고리즘) | `4_greedy_solver.py` | 빠른 근사해, 국소 최적 위험 |
| 5 | GA (유전 알고리즘) | `5_ga_solver.py` | 진화 기반 탐색, 전역 탐색 |
| 6 | AI (강화학습) | `6_ai_solver.py` | 환경 상호작용 기반 학습 |

---

## 📂 파일 구조

```
Portfolio_Manufacturing-Optimization-Algorithms/
├── src/
│   ├── 1_lp_solver.py       # 선형 계획법 (PuLP)
│   ├── 2_ip_solver.py       # 정수 계획법 (PuLP)
│   ├── 3_dp_solver.py       # 동적 계획법
│   ├── 4_greedy_solver.py   # 탐욕 알고리즘
│   ├── 5_ga_solver.py       # 유전 알고리즘
│   ├── 6_ai_solver.py       # 강화학습 기반 최적화
│   ├── environment.py       # 제조 환경 시뮬레이터
│   ├── main_compare.py      # 6개 알고리즘 통합 비교 실행
│   └── visualizer.py        # 결과 시각화
└── results/
    └── comparison_results.png  # 알고리즘별 성능 비교 그래프
```

---

## 🔑 핵심 내용

- **제조 환경 시뮬레이터** (`environment.py`): 수요량, 재료 제약, 기계 용량 등 현실적 제약 조건 모델링
- **알고리즘별 수렴 속도 및 최적화 품질 비교**
- **실행 시간 vs 해의 품질 트레이드오프 분석**

---

## 📊 결과 요약

- LP/IP: 수학적 최적해 보장, 단 소규모 문제에 한정
- DP: 부분 문제 최적화 효과적이나 상태 공간 폭발 문제 존재
- Greedy: 매우 빠르나 최적해 미보장
- GA: 대규모 복잡 문제에서 실용적 근사해 제공
- AI(RL): 반복 경험을 통한 점진적 성능 향상 확인

---

## 📝 사용 기술 스택

- Python, PuLP (LP/IP), NumPy
- 유전 알고리즘 (커스텀 구현)
- 강화학습 (Q-learning 기반)
- Matplotlib (결과 시각화)

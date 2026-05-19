# GitHub 포트폴리오 공개 전 최종 체크리스트

## ✅ 코드 및 테스트

- [ ] 모든 pytest 테스트 통과 (`pytest tests/ -v`)
- [ ] 라벨링 로직이 의도대로 동작하는지 확인
- [ ] 노트북 전체 재실행 완감 (01_eda.ipynb, 02_labeling_logic.ipynb)
- [ ] 라이브러리 import 오류 없음 확인
- [ ] 데이터 경로가 상대 경로로 설정되어 있는지 확인
- [ ] 하드코딩된 절대 경로 제거 확인
- [ ] 테스트에 불필요한 출력/디버그 코드 제거
- [ ] Type hints 및 docstring 검증

## 📄 문서화

- [ ] README.md 전체 검토 및 링크 유효성 확인
- [ ] 배지 이미지 URL 적절성 확인
- [ ] Quick Start 실행 가능성 테스트
- [ ] architecture.md 내용 검토
- [ ] 코드 예시 실행 가능 여부 확인
- [ ] 오타 및 문법 검토

## 📦 의존성 관리

- [ ] requirements.txt 설치 테스트 (`pip install -r requirements.txt`)
- [ ] pyproject.toml이 표준 형식인지 확인
- [ ] Python 3.10 이상에서 실행 가능 여부 확인
- [ ] 불필요한 패키지 제거 및 최소화
- [ ] pytest 설정이 pyproject.toml에 포함되어 있는지 확인

## 🔒 민감 정보 보안

- [ ] 개인 이메일, 전화번호, IP 주소 제거 확인
- [ ] API 키/토큰 제거 확인
- [ ] 실제 회사명/사람 이름 익명화 처리
- [ ] 센서 데이터에 개인정보 포함 여부 확인
- [ ] `.gitignore`에 민감 파일 등록 확인

## 📁 폴더/파일 구조

- [ ] 모든 폴더명 영어로 통일 확인
- [ ] `data/raw` 및 `data/interim` 폴더 생성 확인
- [ ] `.github/workflows/test.yml` 생성 확인
- [ ] `docs/` 폴더 및 이미지 자리표시자 생성 확인
- [ ] 불필요한 파일 및 백업 파일 제거 (`__pycache__`, `*.pyc`, `README_backup.md`)

## 🧪 자동화 테스트

- [ ] GitHub Actions 워크플로우 문법 검증 (test.yml)
- [ ] 최소 Python 3.10에서 테스트 통과 확인
- [ ] 러너 OS별(Ubuntu, Windows) 호환성 확인
- [ ] CI 실패 시 빠른 롤백 계획 수립

## 📝 README 시각 자료

- [ ] 시스템 아키텍처 다이어그램 자리표시자 포함 확인
- [ ] EDA 그래프 스크린샷 자리표시자 포함 확인
- [ ] 라벨 분포 차트 캡션 작성 확인
- [ ] 이상 탐지 타임라인 예시 준비 (로컬 생성 후 업로드)
- [ ] 노트북 실행 화면 스크린샷 준비 계획

## 🎯 포트폴리오 관점

- [ ] 프로젝트가 실제 문제 기반인지 확인
- [ ] Rule-based→ML→Multimodal의 진화 방향이 명확한지 확인
- [ ] 코드 재현성 (Jupyter + 테스트)이 입증되는지 확인
- [ ] "왜" 이 기술 선택을 했는지 설명이 README에 있는지 확인
- [ ] 팀 협업 관점에서 코드 가독성이 우수한지 확인
- [ ] 미래 유지보수를 고려한 구조인지 확인

## 🔗 GitHub 설정

- [ ] 저장소 설명(Description) 작성
- [ ] Topics 추가 (예: `anomaly-detection`, `shared-facility`, `python`)
- [ ] LICENSE 태그 명시 (MIT)
- [ ] GitHub Pages (선택) 설정 고려
- [ ] 저장소 공개(Public) 상태 확인

## ⚡ 성능 및 최적화

- [ ] 노트북 실행 시간 (01_eda, 02_labeling) 확인
- [ ] 데이터 로딩 시간 측정
- [ ] 대규모 데이터셋에서의 메모리 사용량 확인
- [ ] 불필요한 재계산 최소화 여부 점검

## 🎓 학습 자료

- [ ] 새로운 기여자가 따라갈 수 있는 수준의 가이드 확인
- [ ] 각 모듈의 역할이 명확한지 확인
- [ ] 실습 관점에서 단계별 학습이 가능한지 확인
- [ ] 심화 주제로 언급된 부분이 입증되는지 확인

---

## 최종 사전 점검

```bash
# 1. 로컬 테스트 (전체)
pytest tests/ -v

# 2. 노트북 재실행
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_labeling_logic.ipynb

# 3. 의존성 검증
pip install -r requirements.txt

# 4. 코드 정렬 (선택)
black src/

# 5. 린트 검사 (선택)
flake8 src/ tests/

# 6. 최종 폴더 구조 확인
tree /F
```

## 공개 후 체크

- [ ] 저장소 clone 가능 여부 재확인
- [ ] 공개 후 24시간 내 조회 수/star 추적
- [ ] 문제 보고 또는 PR 있을 시 즉시 대응 계획
- [ ] 노트북 렌더링 정상 확인 (GitHub 자체 뷰어)
- [ ] README 링크 정상 작동 확인

---

## 첫 공개 메시지 (선택)

```markdown
# 🎉 공유 물류센터 안전 관리 이상 탐지 시스템

실제 센서 데이터(data_R2.csv)를 기반으로 공유공장/물류센터의 운영 이상을 자동 탐지하는 AI 포트폴리오입니다.

🚀 **핵심 특징**
- Rule-based labeling으로 5개 상태 자동 분류
- 12개 pytest 테스트 전수 통과
- Jupyter 노트북으로 전체 파이프라인 재현 가능
- 지도학습 및 멀티모달 확장 로드맵 포함

📊 **Quick Start**
```bash
pip install -r requirements.txt
pytest tests/ -v
jupyter notebook notebooks/02_labeling_logic.ipynb
```

더 자세한 내용은 [README.md](README.md)를 참고하세요.
```

---

**모든 항목을 확인한 후 "Release Ready" 상태로 최종 판단하세요.**


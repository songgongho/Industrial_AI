# 🚀 PressXAI 대시보드 - 한글 사용 설명서

**최종 업데이트**: 2026년 5월 26일  
**언어**: 한글 100% 지원  
**상태**: ✅ 완료

---

## 📌 가장 간단한 사용 방법 (추천)

### **방법 1: 배치 파일 더블클릭 (가장 권장) ✨**

프로젝트 폴더에서 **`run_dashboard.bat`** 파일을 더블클릭하면:

```
1️⃣ 자동으로 가상환경 활성화됨
2️⃣ Streamlit 필요 패키지 자동 확인
3️⃣ 대시보드 서버 자동 시작
4️⃣ 브라우저에서 http://localhost:8501 자동 오픈
5️⃣ 사용자는 즉시 대시보드 이용 시작
```

**특징:**
- ✅ 가장 간단함 (더블클릭만 하면 됨)
- ✅ Windows 완벽 호환
- ✅ 모든 설정이 자동으로 처리됨
- ✅ 한글 메시지로 상태 표시

---

## 🔧 다른 실행 방법들

### **방법 2: Python 스크립트 더블클릭**

프로젝트 폴더에서 **`run_dashboard.py`** 파일을 더블클릭

```python
# 자동으로 수행되는 작업:
# 1. Streamlit 설치 여부 확인
# 2. 가상환경 확인 (필요한 경우)
# 3. Streamlit 서버 시작
# 4. http://localhost:8501 자동 오픈
```

**특징:**
- ✅ Python을 직접 실행 (배치 파일보다 유연함)
- ✅ 상세한 한글 메시지로 진행 상황 표시
- ✅ Mac/Linux도 작동

---

### **방법 3: 터미널에서 직접 실행**

**Windows PowerShell 또는 CMD:**

```powershell
# 프로젝트 폴더로 이동
cd E:\2026-1학기\Industrial_AI\Portfolio_pcb-lamination-press-defect-prediction

# 방법 A: run_dashboard.py 실행
python run_dashboard.py

# 방법 B: 직접 Streamlit 실행
streamlit run app/streamlit_app.py
```

**Mac/Linux:**

```bash
cd ~/Industrial_AI/Portfolio_pcb-lamination-press-defect-prediction
python run_dashboard.py
```

---

## 📊 대시보드의 탭 설명 (모두 한글)

대시보드를 열면 7개의 탭이 나타납니다:

| 탭 | 기능 | 설명 |
|-----|------|------|
| 📋 **Overview** | 프로젝트 상황판 | 진척도, 마일스톤, 데이터 요청 항목, 최신 KPI |
| 📥 **Data** | 데이터 관리 | 데모 데이터 생성, CSV 업로드, 미리보기 |
| 🎓 **Train** | 모델 학습 | 하이퍼파라미터 조정, 학습 실행, 과정 시각화 |
| 🔮 **Predict** | 예측 결과 | 불량 예측, 성능 메트릭, 혼동 매트릭스 |
| 💡 **Explain** | 설명 가능성 | SHAP 값, Attention 가중치 시각화 |
| 🔗 **Causal** | 인과관계 분석 | 센서 간 인과 관계 그래프, 상위 에지 |
| 📄 **Report** | 리포트 | 학습된 모델 성능 리포트 다운로드 |

---

## ⏹️ 서버 중지하기

### **방법 1: 콘솔 창 닫기**
- 검은색 콘솔 창을 그냥 닫으면 서버가 중지됩니다

### **방법 2: Ctrl+C 누르기**
- 콘솔 창이 활성화된 상태에서 **Ctrl+C** 누르면 즉시 중지

**결과:**
```
⏹️  사용자가 서버를 중지했습니다.
대시보드가 종료되었습니다.
```

---

## 🐛 문제 해결

### **"사이트에 연결할 수 없음" 오류**

**원인:** Streamlit 서버가 제대로 시작되지 않았거나 포트 8501이 사용 중임

**해결 방법:**

```powershell
# 1단계: 포트 확인
netstat -ano | findstr :8501

# 결과가 나타나면 이미 다른 프로세스가 사용 중
# 그 프로세스를 종료하고 다시 시도

# 2단계: 패키지 설치 확인
pip install -r requirements.txt

# 3단계: 다시 시도
python run_dashboard.py
```

---

### **"Streamlit이 설치되지 않았습니다" 메시지**

**해결:**
```powershell
# 패키지 설치
pip install streamlit

# 또는 전체 요구사항 설치 (권장)
pip install -r requirements.txt
```

---

### **브라우저가 자동으로 열리지 않음**

**수동으로 접속:**
1. 브라우저를 직접 열기 (Chrome, Edge, Firefox 등)
2. 주소창에 입력: `http://localhost:8501`
3. Enter 누르기

---

## 📱 대시보드 첫 사용 팁

1. **Overview 탭부터 시작**
   - 프로젝트 현황 파악
   - 마일스톤 확인
   - 고객사 데이터 요청 항목 이해

2. **Data 탭에서 데모 데이터 생성**
   - "데모 데이터 생성" 버튼 클릭
   - 샘플 데이터가 자동으로 생성됨

3. **Train 탭에서 학습 시작**
   - 파라미터 조정 (선택)
   - "학습 실행" 버튼 클릭
   - 진행 상황을 실시간으로 확인

4. **Predict 탭에서 결과 확인**
   - 예측 확률, 성능 메트릭 시각화

5. **Explain 탭에서 모델 해석**
   - SHAP 값으로 특성 중요도 확인
   - Attention 가중치로 의사결정 과정 이해

---

## 🎯 주요 기능 요약

### ✨ 모두 한글로 표시됩니다:
- 모든 버튼, 메뉴, 메시지가 한글
- 그래프, 테이블, 설명도 한글
- 사용자 친화적 인터페이스

### 🔄 자동화된 프로세스:
- 가상환경 자동 활성화 (배치 파일 사용 시)
- 필요 패키지 자동 확인
- 브라우저 자동 오픈
- 서버 자동 구동

### 📊 완전한 대시보드:
- 데이터 관리
- 모델 학습
- 예측 수행
- 결과 분석 및 해석
- 리포트 생성

---

## 🚀 빠른 시작 요약

| 단계 | 실행 방법 | 소요 시간 |
|------|---------|---------|
| 1️⃣ | `run_dashboard.bat` 더블클릭 | 3초 |
| 2️⃣ | http://localhost:8501 접속 (자동) | 즉시 |
| 3️⃣ | Overview 탭 확인 | 1분 |
| 4️⃣ | Data 탭에서 데모 데이터 생성 | 2초 |
| 5️⃣ | Train 탭에서 학습 | 30초~2분 |
| 6️⃣ | Predict, Explain 탭 확인 | 1분 |

**총 소요 시간: 약 5분**

---

## 💾 GitHub 저장소

모든 코드와 리소스는 GitHub에 저장되어 있습니다:
- 저장소: https://github.com/songgongho/Industrial_AI
- 마지막 커밋: `632370d` (Korean UI & simplified launcher)

---

## 📞 지원

문제가 발생하면:
1. 이 문서의 "🐛 문제 해결" 섹션 확인
2. GitHub Issues에 보고
3. 프로젝트 README.md 참고

---

**행운을 빕니다! 🎉 대시보드를 즐겨보세요!**


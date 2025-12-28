# 01 - 선형 회귀 (Linear Regression) 📈

선형 회귀의 기초부터 실제 데이터 적용까지 단계별 학습하는 섹션입니다.

## 📚 학습 목표

1. **선형 회귀의 원리** 이해
2. **단순 vs 다중 회귀** 비교
3. **특성 공학**의 효과 검증
4. **회귀 평가 메트릭** 습득

---

## 📁 파일 구성

### 1. `simple_linear_regression.ipynb`
**내용**: 기본 선형 회귀 모델
- 데이터 생성 (난수)
- 모델 훈련 (scikit-learn)
- 성능 평가
- 결과 시각화

**학습 포인트**:
- Normal Equation: β = (X^T X)^-1 X^T y
- 단순선형 회귀 수식: y = mx + b
- R² Score의 의미

**코드 예시**:
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 모델 생성 및 훈련
model = LinearRegression()
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

---

### 2. `california_housing_feat1.ipynb`
**데이터셋**: California Housing (1개 특성)

**특성**: 
- `MedInc` (중위 소득) - 주택 가격과 높은 상관관계

**성능**:
```
MSE:  1.33
RMSE: 1.15
MAE:  0.82
R²:   0.57
```

**해석**:
- R² = 0.57: 모델이 변동의 57%를 설명
- 1개 특성만으로도 중간 수준의 성능 달성 가능

**시각화**:
- 산점도 + 회귀선
- Residuals 플롯
- 실제값 vs 예측값

---

### 3. `california_housing_feat8.ipynb` ⭐ **핵심**
**데이터셋**: California Housing (모든 8개 특성)

**특성 목록**:
```
1. MedInc      - 중위 소득
2. HouseAge    - 주택 연식
3. AveRooms    - 평균 방 개수
4. AveBedrms   - 평균 침실 개수
5. Population  - 지역 인구
6. AveOccup    - 평균 점유율
7. Latitude    - 위도
8. Longitude   - 경도
```

**전처리 과정**:
```python
from sklearn.preprocessing import StandardScaler

# 1. 데이터 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. 모델 훈련
model = LinearRegression()
model.fit(X_train_scaled, y_train)
```

**성능 비교**:
| 메트릭 | 1개 특성 | 8개 특성 | 개선율 |
|--------|---------|---------|--------|
| MSE | 1.33 | 0.73 | **45%↓** |
| R² | 0.57 | 0.74 | **30%↑** |

**핵심 발견**:
- ✅ 특성 수 증가 → 모델 성능 대폭 향상
- ✅ 정규화가 필수 (특성의 스케일 차이)
- ✅ 지리적 정보(위도/경도)의 중요성

---

## 🎓 기본 개념

### 선형 회귀 수식
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

여기서:
- y: 목표 변수 (가격)
- x: 특성 (독립 변수)
- β: 가중치 (계수)
- ε: 오차항
```

### 비용 함수 (Cost Function)
```
J(β) = (1/2m) * Σ(y_pred - y_actual)²
      = (1/2m) * Σ(ŷ - y)²

목표: J(β)를 최소화하는 β 찾기
```

### Normal Equation
```
β = (X^T X)^-1 X^T y

장점: 한 번에 최적 해를 구함
단점: 계산 비용 높음 (O(n³))
```

---

## 📊 평가 메트릭 설명

### 1. Mean Squared Error (MSE)
```python
MSE = (1/m) * Σ(y_pred - y_actual)²
```
- 작을수록 좋음
- 이상치에 민감
- 단위: (목표변수)²

### 2. Root Mean Squared Error (RMSE)
```python
RMSE = √MSE
```
- 원래 단위로 표현
- 직관적 해석 가능

### 3. Mean Absolute Error (MAE)
```python
MAE = (1/m) * Σ|y_pred - y_actual|
```
- 이상치에 덜 민감
- 평균 오류량

### 4. R² Score (결정 계수)
```
R² = 1 - (SS_res / SS_tot)
   = 1 - (Σ(y_pred - y)² / Σ(y_mean - y)²)

범위: 0 ≤ R² ≤ 1
- R² = 1: 완벽한 예측
- R² = 0: 평균값과 동일한 예측
```

---

## 🔧 실행 방법

### 1. 환경 설정
```bash
# requirements.txt에서 필요한 패키지 설치
pip install -r ../requirements.txt
```

### 2. Jupyter 실행
```bash
cd 01_linear_regression
jupyter notebook
```

### 3. 각 노트북 순서대로 실행
1. `simple_linear_regression.ipynb` (기초)
2. `california_housing_feat1.ipynb` (1개 특성)
3. `california_housing_feat8.ipynb` (다중 회귀)

---

## 💡 실습 팁

### 데이터 이해하기
```python
import pandas as pd

# 데이터 로드
data = pd.read_csv('data.csv')

# 기본 통계
print(data.describe())

# 상관관계
print(data.corr())

# 시각화
data.hist()
```

### 모델 성능 향상 방법
1. **특성 추가**: 관련성 높은 특성 추가
2. **정규화**: 특성 스케일링
3. **다항식**: 비선형 관계 포착
4. **정규화**: L1/L2 정규화 적용

---

## 📌 주요 결론

1. **특성의 중요성**
   - 1개 → 8개 특성: 45% 성능 개선
   - 적절한 특성 선택이 모델 성능을 좌우

2. **정규화의 필요성**
   - StandardScaler로 특성 스케일링
   - 수렴 속도 향상 및 안정성 증가

3. **실무 적용**
   - 대부분의 실세계 문제는 다중 회귀 필요
   - 특성 공학이 매우 중요한 단계

---

## 🔗 다음 단계

선형 회귀 이후 학습할 내용:
→ **다항식 회귀** (비선형 관계)
→ **정규화된 회귀** (Ridge, Lasso)
→ **로지스틱 회귀** (분류 문제)


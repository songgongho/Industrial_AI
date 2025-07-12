import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# 1. 데이터 불러오기
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
df = pd.read_csv(url, parse_dates=['Date'], index_col='Date')
ts = df['Temp']

# 2. 정상성 확인 (ADF 테스트)
result = adfuller(ts)
print(f"Before differencing - ADF p-value: {result[1]:.4f}")

# 3. 차분 (1차 차분)
ts_diff = ts.diff().dropna()

# 4. 다시 ADF 테스트
result_diff = adfuller(ts_diff)
print(f"After differencing - ADF p-value: {result_diff[1]:.4f}")

# 5. 시각화
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
ts.plot(title='Original Series')
plt.grid(True)
plt.subplot(1, 2, 2)
ts_diff.plot(title='1st-order Differenced Series')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. 학습/테스트 분할
train_size = int(len(ts) * 0.8)
train, test = ts[:train_size], ts[train_size:]

# 7. ARIMA 모델 학습 (ARIMA(p,d,q) → 여기서는 ARIMA(5,1,2) 사용 예)
model = ARIMA(train, order=(5, 1, 2))  # (p=5, d=1, q=2)
model_fit = model.fit()

# 8. 예측
forecast = model_fit.forecast(steps=len(test))

# 9. 성능 평가
mse = mean_squared_error(test, forecast)
print(f"ARIMA(5,1,2) MSE: {mse:.4f}")

# 10. 시각화
plt.figure(figsize=(10, 5))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, forecast, label='ARIMA Forecast', linestyle='--')
plt.legend()
plt.title('ARIMA(5,1,2) Forecast vs Actual')
plt.grid(True)
plt.show()

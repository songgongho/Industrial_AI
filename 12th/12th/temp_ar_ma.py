import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. 데이터 로드 (서울 기온 - 예시는 공개된 Kaggle 기온 데이터로 대체)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
df = pd.read_csv(url, parse_dates=['Date'], index_col='Date')
df = df['Temp']  # 시계열 Series로 추출

# 2. 시각화
plt.figure(figsize=(10, 4))
df.plot(title='Daily Minimum Temperature in Melbourne')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()

# 3. 학습/테스트 분리
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# 4. AR 모델 학습 및 예측
ar_model = AutoReg(train, lags=10).fit()
ar_pred = ar_model.predict(start=len(train), end=len(df)-1, dynamic=False)

# 5. MA 모델 학습 및 예측 (ARIMA(0,q,0)으로 구현)
ma_model = ARIMA(train, order=(0, 10, 0)).fit()
ma_pred = ma_model.forecast(steps=len(test))

plt.figure(figsize=(12, 5))
plt.plot(test.index, test, label='Actual', color='black')

plt.figure(figsize=(12, 5))
plt.plot(test.index, ar_pred, label='AR(10) Prediction', linestyle='--')

# 6. 결과 시각화
plt.figure(figsize=(12, 5))
plt.plot(test.index, test, label='Actual', color='black')
plt.plot(test.index, ar_pred, label='AR(10) Prediction', linestyle='--')
plt.plot(test.index, ma_pred, label='MA(10) Prediction', linestyle=':')
plt.legend()
plt.title('AR vs MA Forecast')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()

# 7. 성능 평가
ar_mse = mean_squared_error(test, ar_pred)
ma_mse = mean_squared_error(test, ma_pred)

print(f"AR(10) MSE: {ar_mse:.4f}")
print(f"MA(10) MSE: {ma_mse:.4f}")

import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

# 1. 데이터 불러오기
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
df = pd.read_csv(url, parse_dates=['Date'], index_col='Date')
ts = df['Temp']

# 2. train-test 분할
train_size = int(len(ts) * 0.8)
train, test = ts[:train_size], ts[train_size:]

# 3. auto_arima 모델 찾기 (계절성 없다고 가정)
stepwise_model = auto_arima(
    train,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    seasonal=False,  # 비계절성
    d=None,          # 차분도 자동 결정
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

print(f"\nBest model: {stepwise_model.summary()}")

# 4. 예측
n_periods = len(test)
forecast = stepwise_model.predict(n_periods=n_periods)

# 5. 성능 평가
mse = mean_squared_error(test, forecast)
print(f"Auto ARIMA MSE: {mse:.4f}")

# 6. 시각화
plt.figure(figsize=(10, 5))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, forecast, label='Auto ARIMA Forecast', linestyle='--')
plt.legend()
plt.title('Auto ARIMA Forecast vs Actual')
plt.grid(True)
plt.show()

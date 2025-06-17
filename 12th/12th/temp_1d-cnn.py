import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# 1. 데이터 로드 및 정규화
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
df = pd.read_csv(url, parse_dates=['Date'], index_col='Date')
ts = df['Temp'].values.reshape(-1, 1)

scaler = MinMaxScaler()
ts_scaled = scaler.fit_transform(ts)

# 2. 시퀀스 생성
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len]
        y = data[i+seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LEN = 30
X, y = create_sequences(ts_scaled, SEQ_LEN)

# 3. 데이터 분할
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 4. Tensor 변환 및 CNN 입력 형태 조정 (N, C, L)
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)  # (batch, channel=1, seq_len)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 5. 1D CNN 모델 정의
class CNN1DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)      # (N, 32, L-2)
        x = self.relu(x)
        x = self.pool(x)       # (N, 32, 1)
        x = x.squeeze(-1)      # (N, 32)
        x = self.fc(x)         # (N, 1)
        return x

model = CNN1DModel()

# 6. 학습 설정
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 200

# 7. 학습 루프
for epoch in range(EPOCHS):
    model.train()
    output = model(X_train)
    loss = criterion(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 8. 예측
model.eval()
with torch.no_grad():
    preds = model(X_test).numpy()

# 9. 역정규화 및 평가
preds_rescaled = scaler.inverse_transform(preds)
y_test_rescaled = scaler.inverse_transform(y_test.numpy())

# 10. 시각화
plt.figure(figsize=(12, 5))
plt.plot(y_test_rescaled, label='Actual')
plt.plot(preds_rescaled, label='1D-CNN Prediction', linestyle='--')
plt.legend()
plt.title('1D-CNN Forecast vs Actual')
plt.grid(True)
plt.show()

# 11. MSE 출력
mse = mean_squared_error(y_test_rescaled, preds_rescaled)
print(f"1D-CNN MSE: {mse:.4f}")

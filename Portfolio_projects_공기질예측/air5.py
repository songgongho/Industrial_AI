import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 데이터 로드 및 전처리
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'data', 'AirQualityUCI.csv')
if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, sep=';', decimal=',')
df = df.iloc[:, :-2]
df.replace(-200, np.nan, inplace=True)
df.dropna(inplace=True)
df['Datetime'] = pd.to_datetime(
    df['Date'].str.strip() + ' ' + df['Time'].str.strip(),
    format='%d/%m/%Y %H.%M.%S', errors='coerce'
)
df.dropna(subset=['Datetime'], inplace=True)
df.set_index('Datetime', inplace=True)
df.drop(columns=['Date', 'Time'], inplace=True)

# 2. 상관관계 시각화
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1)
plt.title('주요 변수 간 상관관계')
plt.show()

# 3. 시계열 윈도우 생성 및 Dataset 정의
n_steps = 24

def create_sequences(data: np.ndarray, n_steps: int):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

class TimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vars_group: list, n_steps: int):
        arr = df[vars_group].values
        scaler = MinMaxScaler().fit(arr)
        scaled = scaler.transform(arr)
        X, y = create_sequences(scaled, n_steps)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 4. 모델 정의
class CNNModel(nn.Module):
    def __init__(self, n_features, n_out):
        super().__init__()
        self.conv = nn.Conv1d(n_features, 32, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(32, n_out)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class RNNModel(nn.Module):
    def __init__(self, rnn_type, n_features, hidden_size, n_out):
        super().__init__()
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(n_features, hidden_size, batch_first=True)
        else:
            self.rnn = nn.GRU(n_features, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_out)
    def forward(self, x):
        out, h = self.rnn(x)
        h = h[0] if isinstance(h, tuple) else h
        return self.fc(h[-1])

# 5. 학습 및 평가 함수
epochs = 20
batch_size = 32
learning_rate = 1e-3

def train_and_evaluate(vars_group: list, label: str):
    dataset = TimeSeriesDataset(df, vars_group, n_steps)
    train_size = int(len(dataset) * 0.8)
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    n_features = len(vars_group)
    n_out = len(vars_group)
    models = {
        '1D-CNN': CNNModel(n_features, n_out),
        'LSTM': RNNModel('LSTM', n_features, 64, n_out),
        'GRU': RNNModel('GRU', n_features, 64, n_out)
    }
    results = {}
    for name, model in models.items():
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        # Training
        for _ in range(epochs):
            model.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(Xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
        # Evaluation
        model.eval()
        all_preds, all_truths = [], []
        with torch.no_grad():
            for Xb, yb in test_loader:
                preds = model(Xb.to(device)).cpu().numpy()
                all_preds.append(preds)
                all_truths.append(yb.numpy())
        preds = np.vstack(all_preds)
        truths = np.vstack(all_truths)
        mse = mean_squared_error(truths, preds)
        results[f"{name} - {label}"] = mse
    return results

# 6. 결과 수집
high_vars = ['C6H6(GT)', 'PT08.S2(NMHC)', 'PT08.S5(O3)']
low_vars = ['T', 'RH', 'AH']
res_high = train_and_evaluate(high_vars, 'High Corr')
res_low = train_and_evaluate(low_vars, 'Low Corr')
all_results = {**res_high, **res_low}

# Define model order
model_list = ['1D-CNN', 'LSTM', 'GRU']

# 7. 결과 시각화: High→Low 순서
seq_names = [f"{m} - High Corr" for m in model_list] + [f"{m} - Low Corr" for m in model_list]
seq_vals = [all_results[name] for name in seq_names]
plt.figure(figsize=(8,6))
bars = plt.barh(seq_names, seq_vals, color=['skyblue']*len(model_list) + ['lightgray']*len(model_list))
plt.xlabel('MSE (Mean Squared Error)')
plt.title('모델별 예측 성능 비교 (High→Low)')
for bar, val in zip(bars, seq_vals):
    plt.text(val + 1e-4, bar.get_y()+bar.get_height()/2, f"{val:.4f}", va='center')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 8. 결과 시각화: 모델별 그룹 비교 순서
group_names = []
group_vals = []
for m in model_list:
    group_names.append(f"{m} - High Corr")
    group_vals.append(all_results[f"{m} - High Corr"])
    group_names.append(f"{m} - Low Corr")
    group_vals.append(all_results[f"{m} - Low Corr"])
plt.figure(figsize=(8,6))
bars2 = plt.barh(group_names, group_vals, color=['skyblue' if 'High' in n else 'lightgray' for n in group_names])
plt.xlabel('MSE (Mean Squared Error)')
plt.title('모델별 예측 성능 비교 (모델 그룹 순서)')
for bar, val in zip(bars2, group_vals):
    plt.text(val + 1e-4, bar.get_y()+bar.get_height()/2, f"{val:.4f}", va='center')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 9. 결과 출력
for name, mse in all_results.items():
    print(f"{name}: {mse:.4f}")

# 10. 단위 테스트
if __name__ == '__main__':
    X, y = create_sequences(np.arange(20).reshape(10,2), 4)
    assert X.shape == (6,4,2)
    assert y.shape == (6,2)
    print('create_sequences 테스트 통과')

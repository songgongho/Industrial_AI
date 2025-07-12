import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 생성: 비정상 시계열 (추세가 있는 시계열)
np.random.seed(0)
n = 200
time = np.arange(n)
trend = time * 0.5
noise = np.random.normal(0, 1, n)
non_stationary = trend + noise

# 차분을 통해 정상성 시계열 만들기
differenced = np.diff(non_stationary)

# 시각화
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# 비정상 시계열
axes[0].plot(time, non_stationary, label='비정상 시계열 (추세 있음)', color='red')
axes[0].set_title('비정상 시계열 예시 (추세 포함)')
axes[0].set_ylabel('값')
axes[0].legend()

# 정상 시계열 (차분)
axes[1].plot(time[1:], differenced, label='차분된 시계열 (정상성 있음)', color='blue')
axes[1].set_title('차분 후 정상 시계열 예시')
axes[1].set_xlabel('시간')
axes[1].set_ylabel('값')
axes[1].legend()

plt.tight_layout()
plt.show()

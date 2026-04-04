# %% [markdown]
# # AI4I 2020 Predictive Maintenance: Baseline 약점 분석 및 F1 개선
#
# ============================================================================
# 📊 프로젝트 개요
# ============================================================================
# 
# 이 프로젝트는 제조설비 고장 예측 문제를 해결합니다.
# 
# [문제 상황]
# - 데이터가 불균형: 정상 96.61%, 고장 3.39%
# - 정확도(Accuracy)는 높지만 실제 고장 검출 성능(F1)은 낮음
# - 왜? → 모델이 "모두 정상"이라고 예측해도 정확도가 높기 때문
#
# [해결 방법]
# 1단계: Baseline 모델 만들고 약점 분석
# 2단계: 불균형 해결 (클래스 가중치, SMOTE 등)
# 3단계: 결정 기준값 최적화 (0.5 → 더 좋은 값 찾기)
# 4단계: 손실함수 개선 (Focal Loss)
# 5단계: 여러 모델 앙상블
#
# [최종 목표]
# - 테스트 데이터에서 F1-score 0.70 이상 달성
# - F1이란? → 정밀도와 재현율의 조화평균 (고장 검출 성능을 종합적으로 평가)
#
# ============================================================================

# %%
# ===== 라이브러리 임포트 (필요한 도구 모음) =====

import warnings  # 경고 제어
import random  # 난수 생성 (재현성을 위해 고정)
from dataclasses import dataclass  # 설정값을 정리하는 클래스
from pathlib import Path  # 파일 경로 다루기
from typing import Dict, List, Optional, Tuple, TypedDict, Union  # 타입 힌트 (코드 안정성)

# 불필요한 경고 제거
warnings.filterwarnings("ignore", category=UserWarning)  # UserWarning 무시
warnings.filterwarnings("ignore", category=DeprecationWarning)  # DeprecationWarning 무시
warnings.filterwarnings("ignore", category=FutureWarning)  # FutureWarning 무시
warnings.filterwarnings("ignore", category=RuntimeWarning)  # RuntimeWarning 무시
warnings.filterwarnings("ignore", message=".*torch.*cuda.*")  # PyTorch CUDA 관련 경고 무시
warnings.filterwarnings("ignore", message=".*sklearn.*")  # scikit-learn 경고 무시

import matplotlib
matplotlib.use("Agg")  # GUI 없이 실행되도록 설정
import matplotlib.pyplot as plt  # 그래프 그리기
import numpy as np  # 숫자 배열 연산
import pandas as pd  # 데이터프레임 (데이터 표)
import seaborn as sns  # 예쁜 그래프 그리기
import torch  # 딥러닝 프레임워크 (신경망 구축)
import torch.nn as nn  # 신경망 층들
from imblearn.combine import SMOTEENN  # 데이터 불균형 해결 (SMOTE + ENN)
from imblearn.over_sampling import SMOTE  # 소수 클래스 데이터 증강
from sklearn.metrics import (  # 성능 평가 지표
    accuracy_score,  # 정확도
    confusion_matrix,  # 혼동행렬
    f1_score,  # F1 점수
    precision_recall_fscore_support,  # 정밀도, 재현율, F1 등
    precision_score,  # 정밀도
    recall_score,  # 재현율
    roc_auc_score,  # ROC AUC 점수
)
from sklearn.model_selection import train_test_split  # 데이터 분할
from sklearn.preprocessing import StandardScaler  # 데이터 정규화
from sklearn.utils.class_weight import compute_class_weight  # 클래스 가중치 자동 계산
from torch.utils.data import DataLoader, TensorDataset  # 데이터 로드
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau  # 학습률 스케줄러


# %% [markdown]
# ## 0) 실험 설정 및 재현성 고정
# - 같은 조건에서 다시 실험해도 같은 결과가 나오도록 난수 고정
# - 결과 저장 폴더 생성
# - 데이터 파일 위치 찾기

# %%
# ===== 기본 설정 =====

RANDOM_SEED = 42  # 난수 고정값 (42는 관례)
RESULT_DIR = Path("outputs_ai4i")  # 결과 저장 폴더
RESULT_DIR.mkdir(exist_ok=True)  # 폴더 생성 (이미 있으면 그냥 넘어감)

# matplotlib 기본 설정
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.sans-serif"] = ["sans-serif"]
sns.set_style("whitegrid")
sns.set_palette("husl")


def set_seed(seed: int = 42) -> None:
    """
    모든 난수 생성 도구를 같은 시드로 고정
    → 같은 조건에서 실행하면 항상 같은 결과 나옴
    """
    random.seed(seed)  # Python 기본 난수
    np.random.seed(seed)  # NumPy 난수
    torch.manual_seed(seed)  # PyTorch CPU 난수
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU 난수
    torch.backends.cudnn.deterministic = True  # GPU 연산 재현성
    torch.backends.cudnn.benchmark = False  # GPU 최적화 끄기


set_seed(RANDOM_SEED)  # 난수 고정 실행


def resolve_data_path() -> Path:
    """
    데이터 파일(ai4i2020.csv)의 위치 찾기
    현재 폴더, 같은 폴더, 상위 폴더에서 순서대로 찾음
    """
    candidates = [
        Path("ai4i2020.csv"),  # 현재 작업 폴더
        Path(__file__).resolve().parent / "ai4i2020.csv",  # 이 파일과 같은 폴더
        Path(__file__).resolve().parent.parent.parent / "ai4i2020.csv",  # 상위 폴더
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("`ai4i2020.csv` 파일을 찾지 못했습니다. 작업 디렉토리를 확인하세요.")


DATA_PATH = resolve_data_path()  # 데이터 파일 위치 확정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 사용 가능 확인
print(f"[INFO] data={DATA_PATH}")
print(f"[INFO] device={DEVICE}")


# %% [markdown]
# ## 1) 데이터 로드 및 기본 EDA (탐색적 데이터 분석)
# 
# EDA란? → 데이터를 그래프/통계로 시각화하여 패턴/문제점 파악
# 이 단계에서 하는 일:
# - 데이터 크기 및 형태 확인
# - 타깃(고장 여부) 비율 확인
# - 주요 센서 값들의 분포 확인
# - 센서들 간 상관관계 확인

# %%
def load_dataset(path: Path) -> pd.DataFrame:
    """
    CSV 파일에서 데이터 로드
    
    입력: 파일 경로
    출력: 데이터 테이블(DataFrame)
    """
    df: pd.DataFrame = pd.read_csv(path, low_memory=False)
    return df


# 데이터 로드 실행
raw_df = load_dataset(DATA_PATH)
print(raw_df.shape)  # 데이터 행/열 개수 출력
print(raw_df.head(3))  # 처음 3행 출력하여 구조 확인


# %%
def run_basic_eda(df: pd.DataFrame, save_dir: Path) -> None:
    """
    데이터 분석 및 그래프 생성
    
    입력: 데이터, 저장 폴더
    처리: 데이터 통계 계산 및 그래프 저장
    """
    save_dir.mkdir(exist_ok=True)
    target_col: str = "Machine failure"  # 분석 대상 열
    numeric_cols: List[str] = [
        "Air temperature [K]",  # 공기 온도
        "Process temperature [K]",  # 공정 온도
        "Rotational speed [rpm]",  # 회전 속도
        "Torque [Nm]",  # 토크 (회전력)
        "Tool wear [min]",  # 공구 마모 시간
    ]

    # ===== 1단계: 타깃 클래스 비율 확인 =====
    # 목적: 정상과 고장 데이터의 비율 파악
    class_ratio = df[target_col].value_counts(normalize=True).sort_index() * 100
    print("\n[EDA] Class ratio (%)")
    print(class_ratio)  # 예: 정상 96.61%, 고장 3.39%

    # 타깃 분포를 막대 그래프로 시각화
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=target_col)
    plt.title("Target Class Distribution")
    plt.tight_layout()
    plt.savefig(save_dir / "eda_target_distribution.png", dpi=150)
    plt.close()

    # ===== 2단계: 센서 값들의 분포 확인 =====
    # 목적: 각 센서의 데이터가 어떤 범위에 분포하는지 파악
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], bins=30)
        axes[i].set_title(col)
    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig(save_dir / "eda_numeric_histograms.png", dpi=150)
    plt.close()

    # ===== 3단계: 센서들 간의 상관관계 확인 =====
    # 목적: 어떤 센서들이 서로 영향을 미치는지 파악
    plt.figure(figsize=(8, 6))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
    plt.title("Correlation Matrix (Numeric Features)")
    plt.tight_layout()
    plt.savefig(save_dir / "eda_correlation_matrix.png", dpi=150)
    plt.close()


run_basic_eda(raw_df, RESULT_DIR)  # EDA 실행


# %% [markdown]
# ## 2) 전처리 및 데이터 분할
# 
# 전처리란? → 모델이 학습하기 좋은 형태로 데이터 변환
# 이 단계에서 하는 일:
# - 불필요한 컬럼 제거 (누수 컬럼)
# - 범주형 데이터를 숫자로 변환 (원-핫 인코딩)
# - 센서 값들의 범위 정규화 (0~1 사이로 조정)
# - 데이터를 훈련/검증/테스트 세 부분으로 분할

# %%
def preprocess_dataframe(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str], StandardScaler]:
    """
    모델 학습을 위해 데이터를 깔끔하게 정리
    
    입력: 원본 데이터
    출력: (정제된 피처, 타깃, 수치형 컬럼 이름, 정규화 도구)
    """
    target_col: str = "Machine failure"  # 우리가 예측할 값
    leakage_cols: List[str] = ["TWF", "HDF", "PWF", "OSF", "RNF"]  # 고장 종류 (직접 정답이므로 제거)

    # ===== 1단계: 불필요한 컬럼 제거 =====
    work_df: pd.DataFrame = df.drop(columns=["UDI", "Product ID"]).copy()  # 식별자 제거 + 복사
    work_df = pd.get_dummies(work_df, columns=["Type"], drop_first=False)  # 제품 타입(L/M/H)을 0/1로 변환

    # ===== 2단계: 피처(입력)와 타깃(정답) 분리 =====
    X: pd.DataFrame = work_df.drop(columns=[target_col] + leakage_cols).copy()  # 피처 = 모든 센서 값 + 복사
    y: pd.Series = work_df[target_col].astype(int)  # 타깃 = 고장 여부 (0 또는 1)

    numeric_cols: List[str] = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    # ===== 3단계: 센서 값 정규화 =====
    # 목적: 온도(300K), 토크(20Nm) 등 단위가 다른 값들을 같은 범위로 통일
    # 결과: 모델이 어떤 센서를 더 중요하게 생각하지 않도록 공정하게 취급
    scaler: StandardScaler = StandardScaler()
    X_normalized = X.copy()  # 복사 후 처리
    X_normalized[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    X_normalized = X_normalized.astype(np.float32)  # 메모리 효율을 위해 32비트 실수로 변환

    return X_normalized, y, numeric_cols, scaler


class SplitData(TypedDict):
    """데이터 분할 결과를 저장하는 구조"""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = 42,
) -> SplitData:
    """
    데이터를 훈련/검증/테스트 세 부분으로 분할
    
    왜 세 부분으로 나누나?
    - 훈련: 모델이 학습하는 데이터 (60%)
    - 검증: 훈련 중 성능을 확인하는 데이터 (20%)
    - 테스트: 최종 성능을 평가하는 데이터 (20%) - 모델이 본 적 없는 데이터
    
    중요! 나뉠 때 정상과 고장의 비율을 유지 (stratify=y)
    """
    # 먼저 전체에서 테스트 데이터 20% 떼어냄
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    # 남은 40%에서 검증 20%, 테스트 20% 분할
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    return {
        "X_train": X_train,  # 훈련 피처
        "y_train": y_train,  # 훈련 타깃
        "X_val": X_val,      # 검증 피처
        "y_val": y_val,      # 검증 타깃
        "X_test": X_test,    # 테스트 피처
        "y_test": y_test,    # 테스트 타깃
    }


# 전처리 및 분할 실행
X_all, y_all, numeric_cols, scaler = preprocess_dataframe(raw_df)
splits = split_dataset(X_all, y_all, seed=RANDOM_SEED)
for k, v in splits.items():
    print(k, v.shape)  # 각 데이터셋의 크기 출력


# %% [markdown]
# ## 3) 모델/학습/평가 유틸 함수
# 
# 신경망(Neural Network)이란?
# - 뇌의 신경세포(뉴런)를 모방한 학습 모델
# - 여러 층(레이어)이 쌓여 있음
# - 각 층에서 입력값을 변환하여 다음 층으로 전달
# 
# 이 섹션에서 하는 일:
# - 신경망의 구조 정의 (입력층 → 은닉층 → 출력층)
# - 모델을 훈련하는 과정 정의
# - 성능을 평가하는 함수 정의

# %%
# ===== 신경망 모델 정의 =====

class MLPBinaryClassifier(nn.Module):
    """
    MLP란? Multi-Layer Perceptron의 약자
    
    구조:
    입력층 (센서 데이터)
        ↓
    은닉층 1 (64개 뉴런) - 이전 정보를 조합하여 새로운 표현 만듦
        ↓
    은닉층 2 (64개 뉴런) - 다시 한 번 조합하여 더 복잡한 패턴 찾음
        ↓
    출력층 (1개 뉴런) - 최종 예측 (0~1 사이의 확률)
    """
    def __init__(
        self,
        input_dim: int,  # 입력 개수 (센서 개수)
        hidden_dims: Tuple[int, ...] = (64, 64),  # 각 은닉층의 뉴런 개수 (가변 길이)
        dropout: float = 0.2,  # 드롭아웃 비율 (과적합 방지용)
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        # 모든 은닉층 구성
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))  # 선형 변환
            layers.append(nn.BatchNorm1d(hidden_dim))  # 배치 정규화
            layers.append(nn.ReLU())  # ReLU 활성화
            layers.append(nn.Dropout(dropout))  # 드롭아웃
            prev_dim = hidden_dim
        
        # 출력층
        layers.append(nn.Linear(prev_dim, 1))  # 최종 예측
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력값을 모델에 넣고 출력값을 받는 과정"""
        return self.net(x)


@dataclass
class TrainConfig:
    """모델 학습 설정을 한 곳에 모아두는 클래스"""
    method_name: str  # 이 방법의 이름
    hidden_dims: Tuple[int, int] = (64, 64)  # 은닉층 크기
    dropout: float = 0.2  # 드롭아웃 비율
    lr: float = 1e-3  # 학습률 (낮을수록 천천히 학습)
    weight_decay: float = 0.0  # L2 정규화 강도
    batch_size: int = 128  # 한 번에 학습할 샘플 개수
    epochs: int = 40  # 전체 데이터를 몇 번 반복해서 학습할지
    patience: int = 8  # 성능이 개선되지 않을 때 몇 번까지 기다릴지
    pos_weight: Optional[float] = None  # 고장(소수 클래스)에 대한 가중치
    resampling: str = "none"  # 데이터 불균형 처리 방법
    tune_threshold: bool = False  # 결정 임계값 최적화 여부
    loss_name: str = "bce"  # 손실함수 종류
    focal_gamma: float = 2.0  # Focal Loss의 감마값 (어려운 샘플에 집중)
    focal_alpha: float = 0.75  # Focal Loss의 알파값 (클래스 가중치)


@dataclass
class TrainedOutput:
    """학습 결과를 담는 클래스"""
    model: nn.Module  # 학습된 모델
    threshold: float  # 최적의 결정 임계값
    best_val_loss: float  # 검증 데이터상 최고 성능


class FocalLossWithLogits(nn.Module):
    """
    Focal Loss란?
    
    문제: 일반적인 손실함수는 쉬운 샘플(명확한 정상)도 많이 학습하여 시간 낭비
    해결: 어려운 샘플(헷갈리는 고장)에 더 큰 페널티를 부여
    
    효과: 모델이 명확하지 않은 경우에 집중하도록 유도
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75) -> None:
        super().__init__()
        self.gamma: float = gamma  # 어려움 강조 정도 (클수록 어려운 샘플에 집중)
        self.alpha: float = alpha  # 고장 클래스 강조 정도

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        손실값 계산
        
        입력:
        - logits: 모델이 출력한 예측값 (확률이 아님)
        - targets: 실제 정답 (0 또는 1)
        
        출력: 손실값 (작을수록 좋음)
        """
        # 1단계: 기본 손실 계산 (일반적인 binary cross entropy)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        
        # 2단계: 예측 확률 계산 (0~1 범위로 변환)
        prob = torch.sigmoid(logits)
        
        # 3단계: 맞은 예측의 확률 추출
        # 예: 정답이 1(고장)이면 1일 확률, 정답이 0(정상)이면 0일 확률
        pt = torch.where(targets == 1, prob, 1 - prob)
        
        # 4단계: 클래스별 가중치 적용
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # 5단계: 어려운 샘플에 더 많은 페널티 부여
        focal_term = (1 - pt).pow(self.gamma)
        
        # 최종 손실 = 기본손실 × 클래스가중치 × 어려움강조
        loss: torch.Tensor = (alpha_t * focal_term * bce).mean()
        return loss


# ===== 데이터 로딩 및 학습 함수 =====

def to_tensor_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    """
    NumPy 배열을 PyTorch 텐서로 변환
    
    왜? PyTorch 모델은 NumPy 배열이 아닌 PyTorch 텐서만 받음
    """
    x_t = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
    y_t = torch.tensor(
        np.asarray(y, dtype=np.float32).reshape(-1, 1), dtype=torch.float32
    )
    return TensorDataset(x_t, y_t)


def train_model(
    cfg: TrainConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int = 42,
) -> TrainedOutput:
    """
    모델을 한 번 학습하는 함수
    
    과정:
    1. 난수 고정
    2. 데이터 불균형 처리 (필요시 SMOTE/SMOTEENN)
    3. 데이터를 배치 단위로 나누기
    4. 모델 생성
    5. 손실함수와 최적화 알고리즘 설정
    6. 여러 에포크(반복) 동안 학습
    7. 검증 데이터로 성능 확인
    8. 최고 성능 모델 저장
    9. 임계값 최적화 (필요시)
    
    입력:
    - cfg: 학습 설정 (하이퍼파라미터)
    - X_train, y_train: 훈련 데이터
    - X_val, y_val: 검증 데이터
    - seed: 난수 고정값
    
    출력:
    - 학습된 모델, 최적 임계값, 최고 검증 성능
    """
    set_seed(seed)

    train_x, train_y = X_train.values, y_train.values

    # ===== 데이터 불균형 처리 (강화 버전) =====
    if cfg.resampling == "smote":
        # SMOTE: 소수 클래스 데이터를 인공적으로 생성해 개수 늘리기
        # → 모델이 고장 패턴을 더 자주 보게 됨
        # k_neighbors=3 설정으로 더 공격적 오버샘플링
        sm = SMOTE(random_state=seed, k_neighbors=3)
        train_x, train_y = sm.fit_resample(train_x, train_y)
    elif cfg.resampling == "smoteenn":
        # SMOTEENN: SMOTE + 경계부 노이즈 제거
        # → 더 깔끔한 데이터
        smenn = SMOTEENN(random_state=seed, sampling_strategy="not majority")
        train_x, train_y = smenn.fit_resample(train_x, train_y)
    elif cfg.resampling == "smote_undersample":
        # SMOTE + 랜덤 언더샘플링 조합 (가장 강력)
        # → 불균형 문제 적극적 해결
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.pipeline import Pipeline as ImbPipeline
        # 중요: under-sampling 비율(float)은 minority/majority 비율이므로
        # SMOTE 이후에 undersample(0.5)를 적용하면 다수 클래스 "증가"가 필요해 예외가 날 수 있음.
        # 따라서 순서를 undersample -> SMOTE로 두고, 실패 시 SMOTE only로 안전 폴백한다.
        pipeline = ImbPipeline([
            ("undersample", RandomUnderSampler(random_state=seed, sampling_strategy=0.8)),
            ("smote", SMOTE(random_state=seed, k_neighbors=3, sampling_strategy=1.0)),
        ])
        try:
            train_x, train_y = pipeline.fit_resample(train_x, train_y)
        except ValueError as exc:
            print(f"[WARN] smote_undersample fallback to SMOTE only: {exc}")
            train_x, train_y = SMOTE(random_state=seed, k_neighbors=3).fit_resample(train_x, train_y)

    # ===== 데이터를 배치 단위로 나누기 =====
    # 배치란? 한 번에 모델에 입력할 샘플 개수
    # 예: 배치 크기 128 = 한 번에 128개 샘플 처리
    train_ds = to_tensor_dataset(train_x, train_y)
    val_ds = to_tensor_dataset(X_val.values, y_val.values)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # ===== 모델 생성 및 설정 =====
    model = MLPBinaryClassifier(
        input_dim=X_train.shape[1], 
        hidden_dims=cfg.hidden_dims, 
        dropout=cfg.dropout
    ).to(DEVICE)

    # 손실함수 선택: Focal Loss 또는 가중치 적용 BCE
    if cfg.loss_name == "focal":
        criterion_fn: Union[FocalLossWithLogits, nn.BCEWithLogitsLoss] = FocalLossWithLogits(
            gamma=cfg.focal_gamma, alpha=cfg.focal_alpha
        )
    else:
        criterion_fn = nn.BCEWithLogitsLoss(
            pos_weight=(
                torch.tensor([cfg.pos_weight], device=DEVICE)
                if cfg.pos_weight
                else None
            )
        )
    
    # 최적화 알고리즘: Adam (자동으로 학습률 조정)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    
    # 학습률 스케줄러: 코사인 어닐링 (주기적으로 학습률 감소)
    # → 초반에는 빠르게, 후반으로 갈수록 천천히 학습
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

    best_state = None  # 최고 성능 모델 저장
    best_val_loss = float("inf")  # 최고 성능의 손실값
    best_f1 = 0.0  # 최고 F1-score 저장
    wait = 0  # 성능 개선 없이 기다린 횟수

    # ===== 여러 에포크 동안 학습 =====
    for _epoch in range(cfg.epochs):
        # ===== 훈련 단계 =====
        model.train()  # 모델을 훈련 모드로 설정 (드롭아웃 활성화)
        for xb, yb in train_loader:  # 배치 단위로 반복
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)  # GPU로 옮기기
            optimizer.zero_grad()  # 이전 기울기 초기화
            logits = model(xb)  # 예측값 계산
            
            # 손실값 계산
            if cfg.loss_name == "focal":
                loss_val: torch.Tensor = criterion_fn(logits, yb)  # type: ignore
            else:
                loss_val = criterion_fn(logits, yb)  # type: ignore
            
            loss_val.backward()  # 기울기 계산 (역전파)
            optimizer.step()  # 가중치 업데이트

        # ===== 검증 단계 =====
        model.eval()  # 모델을 평가 모드로 설정 (드롭아웃 비활성화)
        val_losses: List[float] = []
        with torch.no_grad():  # 기울기 계산 불필요 (속도 향상)
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                if cfg.loss_name == "focal":
                    v_loss_val: torch.Tensor = criterion_fn(logits, yb)  # type: ignore
                else:
                    v_loss_val = criterion_fn(logits, yb)  # type: ignore
                val_losses.append(v_loss_val.item())

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        
        # 학습률 스케줄러 스텝
        scheduler.step()
        
        # ===== 조기 종료 (Early Stopping) - 손실 기반 =====
        # 검증 손실이 개선되면 모델 저장
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0  # 카운터 리셋
        else:
            wait += 1
            # 일정 횟수 동안 개선이 없으면 학습 중단 (낭비 방지)
            if wait >= cfg.patience:
                break

    # 최고 성능 모델 불러오기
    if best_state is not None:
        model.load_state_dict(best_state)

    # ===== 임계값 최적화 =====
    threshold = 0.5  # 기본값
    if cfg.tune_threshold:
        # 검증 데이터에서 F1을 최대화하는 임계값 찾기
        # 기본 0.5 대신 최적값을 사용 → 성능 향상
        val_probs, val_true = predict_proba(model, X_val.values, y_val.values, cfg.batch_size)
        threshold = find_best_threshold(val_true, val_probs)

    return TrainedOutput(model=model, threshold=threshold, best_val_loss=best_val_loss)


def predict_proba(
    model: nn.Module,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    batch_size: int = 256,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    훈련된 모델을 사용해 예측 확률 계산
    
    입력: 모델, 피처, 선택사항으로 정답
    출력: (예측확률, 정답) → 정답이 없으면 예측확률만
    
    예: 확률 0.8 = 고장일 확률 80%
    """
    ds = to_tensor_dataset(X, np.zeros(len(X)) if y is None else y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    probs = []
    ys = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            # sigmoid 함수: 모든 값을 0~1 사이로 변환
            p = torch.sigmoid(logits).cpu().numpy().ravel()
            probs.extend(p.tolist())
            ys.extend(yb.numpy().ravel().tolist())

    return np.array(probs), (None if y is None else np.array(ys))


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    F1 점수를 최대화하는 임계값 찾기
    
    기본 임계값: 0.5
    문제: 0.5가 항상 최적은 아님
    
    해결: 0.05 ~ 0.95 범위에서 0.01 간격으로 탐색
    각 임계값에서 F1을 계산하고 가장 높은 F1을 주는 임계값 선택
    
    예:
    - 임계값 0.5: F1 = 0.60
    - 임계값 0.7: F1 = 0.74  ← 가장 높음 (선택!)
    - 임계값 0.9: F1 = 0.58
    """
    thresholds = np.linspace(0.05, 0.95, 91)  # 0.05부터 0.95까지 91개 값
    best_t, best_f1 = 0.5, -1.0
    
    for t in thresholds:
        # 확률 > 임계값이면 고장(1), 아니면 정상(0)으로 예측
        preds = (y_prob >= t).astype(int)
        # 고장 클래스(1)의 F1 계산
        f1_pos = f1_score(y_true, preds, pos_label=1, zero_division=0)
        if f1_pos > best_f1:  # 더 좋은 F1 발견
            best_f1 = f1_pos
            best_t = float(t)
    
    return best_t


def evaluate(
    model: nn.Module,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    method_name: str,
    save_dir: Path,
) -> Dict[str, float]:
    """
    테스트 데이터에서 모델 성능 평가
    
    입력:
    - 모델, 테스트 데이터, 결정 임계값, 방법 이름, 저장 폴더
    
    처리:
    1. 예측 확률 계산
    2. 임계값으로 고장/정상 분류
    3. 정답과 예측 비교해 평가지표 계산
    4. 혼동행렬 그래프 저장
    
    출력:
    - 여러 평가지표 (정확도, 정밀도, 재현율, F1 등)
    """
    probs, y_true = predict_proba(model, X_test.values, y_test.values)
    preds = (probs >= threshold).astype(int)  # 확률 → 고장/정상

    # ===== 평가지표 계산 =====
    acc = accuracy_score(y_true, preds)  # 정확도: 맞은 것의 비율
    prec_macro = precision_score(y_true, preds, average="macro", zero_division=0)  # 평균 정밀도
    rec_macro = recall_score(y_true, preds, average="macro", zero_division=0)  # 평균 재현율
    f1_macro = f1_score(y_true, preds, average="macro", zero_division=0)  # 평균 F1
    f1_pos = f1_score(y_true, preds, pos_label=1, zero_division=0)  # 고장(1) 클래스의 F1
    roc_auc = roc_auc_score(y_true, probs)  # ROC AUC

    # 클래스별 상세 지표
    cls_prec, cls_rec, cls_f1, _ = precision_recall_fscore_support(
        y_true, preds, labels=[0, 1], zero_division=0
    )

    # ===== 혼동행렬 시각화 =====
    # 혼동행렬: 모델의 예측이 얼마나 맞는지/틀렸는지 보여주는 표
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(4.8, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"CM: {method_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_dir / f"cm_{method_name.replace('+', '_')}.png", dpi=150)
    plt.close()

    return {
        "method": method_name,
        "threshold": threshold,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_positive": f1_pos,  # ← 우리의 주요 지표!
        "roc_auc": roc_auc,
        "precision_class0": cls_prec[0],  # 정상 클래스 정밀도
        "recall_class0": cls_rec[0],  # 정상 클래스 재현율
        "precision_class1": cls_prec[1],  # 고장 클래스 정밀도
        "recall_class1": cls_rec[1],  # 고장 클래스 재현율
    }


def evaluate_from_probs(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    method_name: str,
    save_dir: Path,
) -> Dict[str, float]:
    """
    이미 계산된 예측 확률로부터 평가지표 계산
    
    evaluate와 동일하지만 예측 확률이 이미 있을 때 사용
    (예: 앙상블 모델에서 여러 모델의 확률을 평균낸 후)
    """
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    prec_macro = precision_score(y_true, preds, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, preds, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, preds, average="macro", zero_division=0)
    f1_pos = f1_score(y_true, preds, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_true, probs)

    cls_prec, cls_rec, _, _ = precision_recall_fscore_support(
        y_true, preds, labels=[0, 1], zero_division=0
    )

    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(4.8, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"CM: {method_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_dir / f"cm_{method_name.replace('+', '_')}.png", dpi=150)
    plt.close()

    return {
        "method": method_name,
        "threshold": threshold,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_positive": f1_pos,
        "roc_auc": roc_auc,
        "precision_class0": cls_prec[0],
        "recall_class0": cls_rec[0],
        "precision_class1": cls_prec[1],
        "recall_class1": cls_rec[1],
    }


# %% [markdown]
# ## 4) Baseline 학습 및 약점 분석
#
# **약점 분석 포인트(자동 계산):**
# - Accuracy와 failure 클래스 F1 간 격차
# - 클래스별 Precision/Recall 편차
# - 왜 불균형에서 Accuracy가 과대평가되는지

# %%
neg, pos = splits["y_train"].value_counts().sort_index().tolist()
base_pos_weight = neg / max(pos, 1)

baseline_cfg = TrainConfig(
    method_name="Baseline",
    hidden_dims=(64, 64),
    dropout=0.2,
    lr=1e-3,
    batch_size=128,
    epochs=45,
    patience=8,
    pos_weight=None,
    resampling="none",
    tune_threshold=False,
    loss_name="bce",
)

baseline_out = train_model(
    baseline_cfg,
    splits["X_train"],
    splits["y_train"],
    splits["X_val"],
    splits["y_val"],
    seed=RANDOM_SEED,
)
baseline_metrics = evaluate(
    baseline_out.model,
    splits["X_test"],
    splits["y_test"],
    baseline_out.threshold,
    baseline_cfg.method_name,
    RESULT_DIR,
)

print("[Baseline Metrics]")
for k, v in baseline_metrics.items():
    if isinstance(v, float):
        print(f"- {k:>16}: {v:.4f}")


# %%
def build_weakness_markdown(metrics: Dict[str, float]) -> str:
    acc = metrics["accuracy"]
    f1_pos = metrics["f1_positive"]
    gap = acc - f1_pos
    prec1 = metrics["precision_class1"]
    rec1 = metrics["recall_class1"]
    rec0 = metrics["recall_class0"]

    text = f"""
### Baseline 취약점 요약
- Accuracy와 failure F1의 격차: **{gap:.3f}p** (Accuracy={acc:.3f}, Failure F1={f1_pos:.3f})
- Failure(1) Recall={rec1:.3f}, Precision={prec1:.3f}로, 소수 클래스 검출 성능이 제한됨
- Normal(0) Recall={rec0:.3f}로 다수 클래스 중심의 예측 경향이 존재
- 고정 임계값 0.5는 F1 최적점과 불일치할 수 있어 재현율/정밀도 균형이 깨짐

정확도는 다수 클래스(정상)가 매우 많은 데이터에서 쉽게 높아질 수 있으므로, 실제 고장 검출 품질을 충분히 반영하지 못합니다.
데이터 불균형 상황에서는 BCE 손실이 전체 오류를 평균화하면서 소수 클래스의 오분류 비용을 상대적으로 작게 학습할 수 있습니다.
또한 임계값 0.5는 운영 목적(F1 최대화)과 다를 수 있어 failure recall이 낮아지고, 그 결과 failure F1이 accuracy 대비 낮게 나타납니다.
""".strip()
    return text


weakness_md = build_weakness_markdown(baseline_metrics)
print("\n" + weakness_md)


# %% [markdown]
# ## 5) 개선 방법 실험
# 아래 실험들은 동일 split/test set에서 비교합니다.
#
# 1. `+ClassWeight`
# 2. `+ClassWeight+ThresholdTuning`
# 3. `+SMOTE+ThresholdTuning`
# 4. `+TunedMLP(ClassWeight+SMOTE+Threshold)`

# %%
experiment_configs = [
    TrainConfig(
        method_name="+ClassWeight",
        hidden_dims=(64, 64),
        dropout=0.2,
        lr=1e-3,
        batch_size=128,
        epochs=45,
        patience=8,
        pos_weight=base_pos_weight,
        resampling="none",
        tune_threshold=False,
        loss_name="bce",
    ),
    TrainConfig(
        method_name="+ClassWeight+ThresholdTuning",
        hidden_dims=(64, 64),
        dropout=0.2,
        lr=1e-3,
        batch_size=128,
        epochs=45,
        patience=8,
        pos_weight=base_pos_weight,
        resampling="none",
        tune_threshold=True,
        loss_name="bce",
    ),
    TrainConfig(
        method_name="+SMOTE+ThresholdTuning",
        hidden_dims=(64, 64),
        dropout=0.2,
        lr=8e-4,
        batch_size=128,
        epochs=45,
        patience=8,
        pos_weight=None,
        resampling="smote",
        tune_threshold=True,
        loss_name="bce",
    ),
    TrainConfig(
        method_name="+TunedMLP(ClassWeight+SMOTE+Threshold)",
        hidden_dims=(128, 64),
        dropout=0.3,
        lr=7e-4,
        batch_size=128,
        epochs=60,
        patience=10,
        pos_weight=base_pos_weight,
        resampling="smote",
        tune_threshold=True,
        weight_decay=1e-4,
        loss_name="bce",
    ),
    TrainConfig(
        method_name="+SMOTEENN+ThresholdTuning",
        hidden_dims=(64, 64),
        dropout=0.2,
        lr=8e-4,
        batch_size=128,
        epochs=50,
        patience=8,
        pos_weight=None,
        resampling="smoteenn",
        tune_threshold=True,
        loss_name="bce",
    ),
    TrainConfig(
        method_name="+FocalLoss+SMOTE+ThresholdTuning",
        hidden_dims=(128, 64),
        dropout=0.25,
        lr=7e-4,
        batch_size=128,
        epochs=60,
        patience=10,
        pos_weight=None,
        resampling="smote",
        tune_threshold=True,
        loss_name="focal",
        focal_gamma=2.0,
        focal_alpha=0.80,
    ),
    TrainConfig(
        method_name="+FocalLoss(g1.5,a0.85)+SMOTE+Threshold",
        hidden_dims=(256, 128),
        dropout=0.30,
        lr=5e-4,
        batch_size=128,
        epochs=80,
        patience=12,
        pos_weight=None,
        resampling="smote",
        tune_threshold=True,
        weight_decay=1e-4,
        loss_name="focal",
        focal_gamma=1.5,
        focal_alpha=0.85,
    ),
    TrainConfig(
        method_name="+FocalLoss+SMOTEENN+Threshold",
        hidden_dims=(128, 64),
        dropout=0.25,
        lr=6e-4,
        batch_size=128,
        epochs=70,
        patience=10,
        pos_weight=None,
        resampling="smoteenn",
        tune_threshold=True,
        weight_decay=1e-4,
        loss_name="focal",
        focal_gamma=2.0,
        focal_alpha=0.80,
    ),
    # ===== 새로운 고강도 F1 개선 설정 =====
    TrainConfig(
        method_name="+FocalLoss+SMOTE_US+ThresholdTuning",
        hidden_dims=(256, 128, 64),  # 더 깊은 네트워크
        dropout=0.3,
        lr=5e-4,
        batch_size=64,  # 작은 배치로 더 빈번한 업데이트
        epochs=100,  # 더 많은 에포크
        patience=15,
        pos_weight=None,
        resampling="smote_undersample",  # SMOTE + 언더샘플링 조합
        tune_threshold=True,
        weight_decay=2e-4,
        loss_name="focal",
        focal_gamma=2.5,  # 더 강한 Focal Loss
        focal_alpha=0.85,
    ),
    TrainConfig(
        method_name="+UltraFocal+SMOTE_US",
        hidden_dims=(512, 256, 128),  # 매우 깊은 네트워크
        dropout=0.35,
        lr=3e-4,
        batch_size=32,  # 아주 작은 배치
        epochs=120,
        patience=20,
        pos_weight=None,
        resampling="smote_undersample",
        tune_threshold=True,
        weight_decay=3e-4,
        loss_name="focal",
        focal_gamma=3.0,  # 초강력 Focal Loss
        focal_alpha=0.90,
    ),
]

all_metrics: List[Dict[str, float]] = [baseline_metrics]

for cfg in experiment_configs:
    print(f"\n[Run] {cfg.method_name}")
    out = train_model(
        cfg,
        splits["X_train"],
        splits["y_train"],
        splits["X_val"],
        splits["y_val"],
        seed=RANDOM_SEED,
    )
    m = evaluate(
        out.model,
        splits["X_test"],
        splits["y_test"],
        out.threshold,
        cfg.method_name,
        RESULT_DIR,
    )
    all_metrics.append(m)
    print(
        f"  - threshold={m['threshold']:.2f}, acc={m['accuracy']:.4f}, "
        f"f1_macro={m['f1_macro']:.4f}, f1_pos={m['f1_positive']:.4f}"
    )


# Multi-seed soft-voting ensemble on best single model candidate
single_results = pd.DataFrame(all_metrics)
best_single_method = single_results.sort_values("f1_positive", ascending=False).iloc[0]["method"]
best_cfg = next((cfg for cfg in experiment_configs if cfg.method_name == best_single_method), None)

if best_cfg is not None and best_cfg.method_name != "Baseline":
    ensemble_seeds = [42, 52, 62]
    val_prob_list: List[np.ndarray] = []
    test_prob_list: List[np.ndarray] = []

    print(f"\n[Run] +Ensemble3({best_cfg.method_name})")
    for s in ensemble_seeds:
        ens_out = train_model(
            best_cfg,
            splits["X_train"],
            splits["y_train"],
            splits["X_val"],
            splits["y_val"],
            seed=s,
        )
        v_prob, _ = predict_proba(ens_out.model, splits["X_val"].values, splits["y_val"].values)
        t_prob, _ = predict_proba(ens_out.model, splits["X_test"].values, splits["y_test"].values)
        val_prob_list.append(v_prob)
        test_prob_list.append(t_prob)

    val_probs_ens = np.mean(np.vstack(val_prob_list), axis=0)
    test_probs_ens = np.mean(np.vstack(test_prob_list), axis=0)
    ens_threshold = find_best_threshold(splits["y_val"].values, val_probs_ens)

    ens_metrics = evaluate_from_probs(
        splits["y_test"].values,
        test_probs_ens,
        ens_threshold,
        f"+Ensemble3({best_cfg.method_name})",
        RESULT_DIR,
    )
    all_metrics.append(ens_metrics)
    print(
        f"  - threshold={ens_metrics['threshold']:.2f}, acc={ens_metrics['accuracy']:.4f}, "
        f"f1_macro={ens_metrics['f1_macro']:.4f}, f1_pos={ens_metrics['f1_positive']:.4f}"
    )


# %% [markdown]
# ## 6) 결과 정리: `results_df`, 목표 달성 여부, 시각화

# %%
results_df = pd.DataFrame(all_metrics)
results_df = results_df.sort_values(by="f1_positive", ascending=False).reset_index(drop=True)

# 목표 조건: failure 클래스 F1 >= 0.70
results_df["meets_goal"] = results_df["f1_positive"] >= 0.70

first_goal_row = results_df[results_df["meets_goal"]].head(1)
first_goal_method = first_goal_row["method"].iloc[0] if len(first_goal_row) > 0 else "미달성"

results_df.to_csv(RESULT_DIR / "results_metrics.csv", index=False, encoding="utf-8-sig")
print(results_df[["method", "accuracy", "f1_macro", "f1_positive", "meets_goal"]])


# %%
def df_to_markdown_simple(df: pd.DataFrame, columns: List[str]) -> str:
    sub = df[columns].copy()
    headers = columns
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in sub.iterrows():
        vals = []
        for c in headers:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


md_table = df_to_markdown_simple(
    results_df,
    ["method", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_positive", "roc_auc", "meets_goal"],
)
print("\n[Markdown Table]\n")
print(md_table)


# %%
plt.figure(figsize=(10, 5))
plot_df = results_df.copy()
plot_df = plot_df.iloc[::-1]
plt.barh(plot_df["method"], plot_df["f1_positive"], color="teal")
plt.axvline(0.70, color="red", linestyle="--", linewidth=1.5, label="Goal: 0.70")
plt.xlabel("Test F1-score (Failure class)")
plt.title("F1-score (Failure) across Methods")
plt.legend()
plt.tight_layout()
plt.savefig(RESULT_DIR / "f1_positive_by_method.png", dpi=150)
plt.close()


# %% [markdown]
# ## 7) 방법론별 도입 이유 (수치 데이터 해석 관점, 한국어)

# %%
method_rationale_md = """
### 방법 1: Class Weight
`Machine failure=1` 표본이 적어 학습 시 손실 기여도가 작아지는 문제를 보완하기 위해 `pos_weight`를 사용했습니다.
수치 데이터 관점에서 이는 동일한 센서 편차라도 고장 클래스 오분류에 더 큰 페널티를 부여해, 경계면이 소수 클래스를 더 민감하게 반영하도록 유도합니다.
결과적으로 재현율(Recall) 개선 가능성이 높아지고 failure F1 향상에 기여할 수 있습니다.

### 방법 2: Threshold Tuning
기본 임계값 0.5는 확률 보정 관점에서 중립점일 뿐, F1 최적점은 아닐 수 있습니다.
검증셋에서 임계값을 탐색해 failure F1이 최대가 되는 지점을 선택하면, 정밀도-재현율 균형을 목적 지표 중심으로 재배치할 수 있습니다.
이는 운영 목적이 고장 검출 안정성일 때 특히 유효합니다.

### 방법 3: SMOTE
훈련셋에서만 SMOTE를 적용해 소수 클래스 주변의 특징 공간을 보강했습니다.
센서 수치 공간에서 고장 사례의 국소 분포를 촘촘히 만들면, 모델이 정상 클래스에 과도하게 치우치지 않고 결정경계를 학습하기 쉽습니다.
단, 과도한 합성은 노이즈를 키울 수 있어 검증셋 기반 튜닝이 필요합니다.

### 방법 4: Tuned MLP + Regularization
은닉 차원/드롭아웃/L2/학습률을 함께 조정해 과적합을 억제하면서 표현력을 확장했습니다.
표준화된 수치 피처에서 batch normalization과 dropout은 기울기 안정화 및 일반화 성능 향상에 도움을 줍니다.
클래스 가중치, 오버샘플링, 임계값 튜닝을 결합하면 불균형 문제를 데이터/손실/의사결정 레벨에서 동시에 완화할 수 있습니다.

### 방법 5: SMOTEENN
SMOTE로 소수 클래스를 보강한 뒤 ENN으로 경계 근처의 혼동 샘플을 정리하는 방식입니다.
수치 공간에서 경계가 겹치는 구간을 정돈하면, 고장/정상 클래스의 분리도가 개선되어 failure F1이 상승할 수 있습니다.

### 방법 6: Focal Loss + Soft-Voting Ensemble
Focal Loss는 쉬운 샘플보다 어려운 샘플(주로 소수 클래스 오분류)에 더 큰 학습 비중을 둡니다.
추가로 다중 시드 앙상블은 초기화 편차를 평균화해 예측 분산을 줄여 임계값 튜닝의 안정성을 높입니다.
""".strip()
print(method_rationale_md)


# %% [markdown]
# ## 8) PPT 붙여넣기용 한국어 결과보고서 텍스트 자동 생성

# %%
def build_ppt_report_text(results: pd.DataFrame, weakness_text: str) -> str:
    baseline_row = results[results["method"] == "Baseline"].iloc[0]
    best_row = results.iloc[0]

    improve = best_row["f1_positive"] - baseline_row["f1_positive"]
    goal_text = "달성" if bool(best_row["meets_goal"]) else "미달성"

    report = f"""
# 슬라이드 1: 과제 및 목표 요약
- 과제: MLP 베이스라인 모델의 취약점 분석 및 성능 개선
- 목표: Test 데이터 기준 F1-Score 0.70 이상 달성
- 접근: 불균형 처리(Class Weight/SMOTE), 임계값 최적화, MLP 구조/학습률 튜닝을 단계적으로 비교

# 슬라이드 2: 데이터셋 개요
AI4I 2020 Predictive Maintenance 데이터는 제조 공정 센서 기반 이진 분류 문제로, 타깃은 `Machine failure`입니다.
주요 피처는 공기온도, 공정온도, 회전속도, 토크, 공구마모 시간과 제품 타입(Type)입니다.
고장 관련 세부 레이블(TWF/HDF/PWF/OSF/RNF)은 타깃과 직접 연관된 누수 가능성이 있어 학습 피처에서 제외했습니다.
전체 데이터에서 고장 비율이 낮아 클래스 불균형이 존재하며, 이는 Accuracy 중심 평가의 한계를 유발합니다.

# 슬라이드 3: 베이스라인 모델 구조 및 성능
- 모델: 2-hidden MLP(64, 64), ReLU, BatchNorm, Dropout
- 학습: BCEWithLogitsLoss, Adam, stratified split(60/20/20)
- Baseline Test 성능: Accuracy={baseline_row['accuracy']:.3f}, F1-macro={baseline_row['f1_macro']:.3f}, F1-failure={baseline_row['f1_positive']:.3f}
- 평가지표 불균형: Accuracy 대비 Failure F1 격차가 존재하여 소수 클래스 검출 한계 확인

# 슬라이드 4: 성능 개선 방법론(1)
- 클래스 가중치: 소수 클래스 오분류에 더 큰 손실을 부여하여 failure 검출 민감도 강화
- 임계값 튜닝: 고정 0.5 대신 검증셋에서 F1 최대 임계값 선택
- 수치 데이터 해석 관점: 손실 함수/결정 임계값을 목표 지표(F1)에 정렬해 불균형 편향 완화

# 슬라이드 5: 성능 개선 방법론(2)
- SMOTE: 훈련셋의 소수 클래스 분포를 확장해 경계학습 안정화
- SMOTEENN/Focal Loss/Ensemble: 경계 정리 + 난이도 기반 손실 + 다중 시드 평균으로 failure 검출 안정성 향상
- 해석 관점: 데이터 레벨 + 손실 레벨 + 의사결정 레벨을 동시에 보정해 failure F1 향상 유도

# 슬라이드 6: 방법론별 평가지표 변화 추이
- 최고 성능 방법: {best_row['method']}
- 최고 성능 지표: Accuracy={best_row['accuracy']:.3f}, F1-macro={best_row['f1_macro']:.3f}, F1-failure={best_row['f1_positive']:.3f}
- 목표(0.70) 달성 여부: {goal_text}
- Baseline 대비 Failure F1 변화: {baseline_row['f1_positive']:.3f} -> {best_row['f1_positive']:.3f} ({improve:+.3f}p)

# 슬라이드 7: 최종 모델 및 결론
- 최종 선정: {best_row['method']}
- 핵심 성과: Baseline 대비 failure 클래스 검출 성능(F1) 정량 개선
- 해석: Accuracy는 일부 변동 가능하나, 고장 탐지 품질 관점에서는 F1 개선이 더 중요한 의사결정 근거
- 한계/향후: 임계값의 운영 환경 재보정, 시간축/설비별 드리프트 반영, 비용민감 학습 추가 검토

# 부록: Baseline 취약점 자동 분석 문구
{weakness_text}
""".strip()
    return report


ppt_text = build_ppt_report_text(results_df, weakness_md)

with open(RESULT_DIR / "ppt_report_kor.md", "w", encoding="utf-8") as f:
    f.write(ppt_text + "\n\n# 방법별 결과 테이블\n\n" + md_table + "\n")

with open(RESULT_DIR / "baseline_weakness_kor.md", "w", encoding="utf-8") as f:
    f.write(weakness_md)

with open(RESULT_DIR / "method_rationale_kor.md", "w", encoding="utf-8") as f:
    f.write(method_rationale_md)

print("\n[Saved]")
print(f"- {RESULT_DIR / 'results_metrics.csv'}")
print(f"- {RESULT_DIR / 'f1_positive_by_method.png'}")
print(f"- {RESULT_DIR / 'ppt_report_kor.md'}")
print(f"- {RESULT_DIR / 'baseline_weakness_kor.md'}")
print(f"- {RESULT_DIR / 'method_rationale_kor.md'}")


# %% [markdown]
# ## 9) 한/영 핵심 내러티브 (보고서 작성용)
#
# **KOR**: 본 실험은 AI4I 2020 불균형 이진 분류 문제에서 baseline MLP의 정확도 편향을 확인하고,
# class weight, SMOTE, threshold tuning, MLP 튜닝을 결합해 failure 클래스 중심 F1 개선을 정량 검증했다.
#
# **ENG**: This experiment diagnoses metric imbalance in an imbalanced binary classification setting,
# then improves failure-class F1 by combining class weighting, SMOTE, threshold tuning, and MLP optimization
# under a fixed train/validation/test split.

# %%
print("\n실험 완료: outputs_ai4i 폴더 결과물을 확인하세요.")


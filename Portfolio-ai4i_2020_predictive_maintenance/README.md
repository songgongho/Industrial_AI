# AI4I 2020 MLP Baseline and F1 Improvement

이 폴더는 `ai4i2020.csv`를 사용해 다음을 수행합니다.
- 베이스라인 MLP 학습 및 평가지표(Accuracy, Precision/Recall/F1, Confusion Matrix)
- 불균형 개선(Class Weight, SMOTE, Threshold Tuning, MLP 튜닝)
- 방법별 성능 비교 테이블/그래프 생성
- PPT 결과보고서용 한국어 마크다운 자동 생성

## 파일 구성
- `ai4i_f1_improvement_notebook.py`: Jupyter 스타일(`# %%`) 실험 코드
- `requirements.txt`: 실행 의존성
- 실행 후 생성: `outputs_ai4i/` (지표 CSV, 그래프, 보고서 md)

## 1) 가상환경 생성 (Windows PowerShell)
```powershell
Set-Location "E:\2026-1학기\5th_practice(화요일)\5th_practice_numerical\ai4i_2020_predictive_maintenance"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 1-1) NVIDIA GPU(PyTorch CUDA) 설정
기본 `requirements.txt` 설치는 CPU 빌드가 잡힐 수 있으므로, GPU 사용 시 아래 명령으로 CUDA 빌드 torch를 설치하세요.

```powershell
Set-Location "E:\2026-1학기\5th_practice(화요일)\5th_practice_numerical\ai4i_2020_predictive_maintenance"
.\.venv\Scripts\Activate.ps1
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

검증 출력에서 `torch.cuda.is_available()`가 `True`이고 GPU 이름이 표시되면 설정 완료입니다.

## 2) 실행
`ai4i2020.csv`가 현재 폴더 또는 상위 워크스페이스 루트에 있으면 자동으로 인식됩니다.

```powershell
python .\ai4i_f1_improvement_notebook.py
```

## 3) 주요 결과물
- `outputs_ai4i/results_metrics.csv`
- `outputs_ai4i/f1_positive_by_method.png`
- `outputs_ai4i/ppt_report_kor.md`
- `outputs_ai4i/baseline_weakness_kor.md`
- `outputs_ai4i/method_rationale_kor.md`

## 참고
- 목표 판정은 `failure F1 >= 0.70` 기준으로 처리합니다.
- 모델 하이퍼파라미터(은닉층, learning rate, epoch 등)는 스크립트의 `TrainConfig`에서 쉽게 변경 가능합니다.


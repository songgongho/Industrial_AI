# Bedrock AI Agent 실습 가이드

이 프로젝트는 `streamlit` + `AWS Bedrock Converse API` + `Lambda Tool` 조합으로 AI Agent 동작을 테스트하기 위한 최소 실습 예제입니다.

## 1) 프로젝트 구성

- `app.py`: Streamlit 기반 멀티-모델 Bedrock Agent UI
- `scripts/preflight_check.py`: AWS 자격증명/Bedrock/Lambda 사전 점검
- `requirements.txt`: Python 의존성
- `main.py`: 로컬 실행 안내용 간단한 헬퍼
- `sample code1.py`: 원본 샘플 파일(참고용)

## 2) 가상환경 생성 (Windows PowerShell)

```powershell
Set-Location "E:\2026-1학기\AI Agent 실습"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3) AWS 인증 및 권한 준비

사전 조건:
- AWS 계정에서 Bedrock 모델 접근 승인(예: `amazon.nova-micro-v1:0`)
- IAM 사용자/역할에 최소한 다음 권한 필요
  - `bedrock:Converse`
  - `bedrock:ConverseStream`
  - `lambda:InvokeFunction`
  - `lambda:GetFunction`
  - `sts:GetCallerIdentity`

AWS CLI로 프로파일을 설정한 경우:

```powershell
aws configure
```

환경변수로 설정할 경우:

```powershell
$env:AWS_ACCESS_KEY_ID="<YOUR_KEY>"
$env:AWS_SECRET_ACCESS_KEY="<YOUR_SECRET>"
$env:AWS_DEFAULT_REGION="us-east-1"
```

## 4) 실습용 환경변수 (선택)

기본값이 이미 코드에 들어있지만, 필요 시 변경 가능합니다.

```powershell
$env:BEDROCK_REGION="us-east-1"
$env:LAMBDA_REGION="us-east-1"
$env:WEATHER_LAMBDA_FUNCTION="GetWeatherFunction"
$env:BEDROCK_MODEL_ID="amazon.nova-micro-v1:0"
```

## 5) 사전 점검

```powershell
python -m scripts.preflight_check
```

성공 기준:
- STS 자격증명 확인 OK
- Bedrock 호출 OK
- Lambda 접근 OK(또는 최소 WARN 원인 파악)

## 6) Streamlit 앱 실행

```powershell
streamlit run app.py
```

브라우저에서 다음 예시 질문을 테스트하세요.
- `뉴욕 시간이랑 런던 날씨 알려줘`
- `서울, 도쿄 현재 시간 비교해줘`
- `오늘 런던 날씨를 알려주고 시간도 같이 표시해줘`

## 7) 트러블슈팅

- `NoCredentialsError`
  - AWS 자격증명 미설정. `aws configure` 또는 환경변수 확인
- `AccessDeniedException`
  - IAM 권한 부족. Bedrock/Lambda 권한 정책 점검
- `ModelNotReadyException` 또는 모델 호출 실패
  - Bedrock 모델 접근 승인 여부, 리전 일치 여부 확인
- Lambda 호출 실패
  - 함수명(`WEATHER_LAMBDA_FUNCTION`)과 리전(`LAMBDA_REGION`) 확인
  - 이 예제는 실패 시 mock 날씨 데이터로 자동 폴백

## 8) 빠른 시작 체크리스트

- [ ] `.venv` 생성 및 활성화
- [ ] `pip install -r requirements.txt`
- [ ] AWS 인증 설정
- [ ] `python -m scripts.preflight_check` 성공
- [ ] `streamlit run app.py` 실행 후 대화 테스트


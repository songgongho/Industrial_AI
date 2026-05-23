# 이상탐지 해석 에이전트 (Copilot 실습)

SECOM 계열 제조 데이터 시나리오를 가정해, 모델 출력(`anomaly_score`, `feature_importance`)을 한국어 보고서로 변환하는 CLI 예제입니다.

## 구성 파일

- `main.py`: Copilot용 프롬프트 주석 + CLI 진입점
- `tools.py`: 입력 검증, 통계 분석, 심각도 분류, 보고서 생성, 선택적 LLM 호출
- `sample_input.json`: 실행 샘플 데이터
- `smoke_test.py`: 최소 동작 확인 스크립트
- `analyze_mdb.py`: Press `.mdb` 로그 파일 일괄 분석 스크립트

## 빠른 실행

```powershell
python -m pip install -r requirements.txt
python main.py --input sample_input.json
```

## 보고서 파일 저장

```powershell
python main.py --input sample_input.json --output report.md
```

## 선택적 LLM 해석

Ollama(로컬) 예시:

```powershell
python main.py --input sample_input.json --use-llm --llm-provider ollama --llm-model llama3.1
```

OpenAI 예시는 프로젝트 루트에 `.openai_api_key` 파일을 두고 실행합니다.

```powershell
python main.py --input sample_input.json --use-llm --llm-provider openai --llm-model gpt-4o-mini
```

## 스모크 테스트

```powershell
python smoke_test.py
```

## Press MDB 직접 분석

`학습대상/Press Profile log` 경로의 `.mdb` 파일을 읽어 이상탐지 보고서를 생성합니다.

```powershell
python analyze_mdb.py
```

실행 후 결과는 `press_mdb_analysis.md`에 저장됩니다.


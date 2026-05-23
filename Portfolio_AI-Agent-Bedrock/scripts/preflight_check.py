import os
import sys

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError


def check_sts(session):
    try:
        sts = session.client("sts")
        identity = sts.get_caller_identity()
        arn = identity.get("Arn", "unknown")
        print(f"[OK] AWS 자격증명 확인: {arn}")
        return True
    except NoCredentialsError:
        print("[FAIL] AWS 자격증명이 없습니다. `aws configure` 또는 환경변수를 설정하세요.")
    except (BotoCoreError, ClientError) as e:
        print(f"[FAIL] STS 호출 실패: {e}")
    return False


def check_bedrock_model(session, region, model_id):
    try:
        client = session.client("bedrock-runtime", region_name=region)
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "안녕"}]}],
            inferenceConfig={"maxTokens": 16},
        )
        output = response.get("output", {}).get("message", {}).get("content", [])
        text = output[0].get("text", "") if output else ""
        print(f"[OK] Bedrock 모델 호출 성공 ({model_id}, {region}) -> {text[:30]}")
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "Unknown")
        msg = e.response.get("Error", {}).get("Message", str(e))
        print(f"[FAIL] Bedrock 호출 실패 ({code}): {msg}")
    except (BotoCoreError, Exception) as e:
        print(f"[FAIL] Bedrock 점검 중 오류: {e}")
    return False


def check_lambda(session, region, function_name):
    try:
        client = session.client("lambda", region_name=region)
        response = client.get_function(FunctionName=function_name)
        arn = response.get("Configuration", {}).get("FunctionArn", "unknown")
        print(f"[OK] Lambda 접근 가능: {arn}")
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "Unknown")
        msg = e.response.get("Error", {}).get("Message", str(e))
        print(f"[WARN] Lambda 점검 실패 ({code}): {msg}")
    except (BotoCoreError, Exception) as e:
        print(f"[WARN] Lambda 점검 중 오류: {e}")
    return False


def main():
    region = os.getenv("BEDROCK_REGION", "us-east-1")
    lambda_region = os.getenv("LAMBDA_REGION", region)
    model_id = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")
    function_name = os.getenv("WEATHER_LAMBDA_FUNCTION", "GetWeatherFunction")

    print("=== Bedrock Agent 사전 점검 ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"BEDROCK_REGION={region}")
    print(f"LAMBDA_REGION={lambda_region}")
    print(f"BEDROCK_MODEL_ID={model_id}")
    print(f"WEATHER_LAMBDA_FUNCTION={function_name}")
    print()

    session = boto3.session.Session()
    ok_sts = check_sts(session)
    ok_bedrock = check_bedrock_model(session, region, model_id) if ok_sts else False
    ok_lambda = check_lambda(session, lambda_region, function_name) if ok_sts else False

    print("\n=== 결과 ===")
    print(f"- 자격증명: {'OK' if ok_sts else 'FAIL'}")
    print(f"- Bedrock: {'OK' if ok_bedrock else 'FAIL'}")
    print(f"- Lambda: {'OK' if ok_lambda else 'WARN/FAIL'}")

    if ok_sts and ok_bedrock:
        print("\n[READY] Streamlit 앱 테스트를 진행할 수 있습니다.")
        return 0

    print("\n[NOT READY] 위 실패 항목을 먼저 해결하세요.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


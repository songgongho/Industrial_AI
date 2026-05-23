import streamlit as st
import boto3
import json
from datetime import datetime
import pytz
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError

# ==========================================
# 1. 로컬 파이썬 Tool 함수 (시간 확인)
# ==========================================
def get_current_time(timezone_str="Asia/Seoul"):
    if not timezone_str:
        timezone_str = "Asia/Seoul"

    tz_mapping = {
        "Seoul": "Asia/Seoul",       "서울": "Asia/Seoul",
        "New_York": "America/New_York", "뉴욕": "America/New_York",
        "London": "Europe/London",   "런던": "Europe/London",
        "Tokyo": "Asia/Tokyo",       "도쿄": "Asia/Tokyo",
    }

    mapped_tz = tz_mapping.get(timezone_str, timezone_str)

    try:
        tz = pytz.timezone(mapped_tz)
        now = datetime.now(tz)
        korean_time_format = f"{now.year}년 {now.month}월 {now.day}일 {now.hour}시 {now.minute}분"
        return json.dumps({"status": "success", "timezone": mapped_tz, "current_time": korean_time_format}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


# ==========================================
# 2. Bedrock Converse API 규격의 도구 명세서
# ==========================================
tool_config = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_current_time",
                "description": "특정 지역(타임존)의 현재 날짜와 시간을 반환합니다.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "timezone": {
                                "type": "string",
                                "description": "타임존 (예: Asia/Seoul, America/New_York, 서울, 뉴욕)"
                            }
                        },
                        "required": ["timezone"]
                    }
                }
            }
        },
        {
            "toolSpec": {
                "name": "invoke_weather_lambda",
                "description": "AWS Lambda를 호출하여 특정 도시의 현재 날씨와 기온을 가져옵니다.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "영문 도시 이름 (예: Seoul, New_York, London)"
                            }
                        },
                        "required": ["city"]
                    }
                }
            }
        }
    ]
}


# ==========================================
# 3. Streamlit 화면 구성
# ==========================================
st.set_page_config(
    page_title="Multi-Model Bedrock 에이전트",
    page_icon="🧠",
    layout="wide"
)

with st.sidebar:
    st.header("⚙️ 모델 설정")

    selected_model = st.selectbox(
        "사용할 LLM을 선택하세요",
        options=[
            "amazon.nova-micro-v1:0",
            "amazon.nova-pro-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
        ],
        index=0,
    )
    st.caption("선택한 모델에 따라 도구 호출의 정확도와 추론 능력이 달라집니다.")

    st.divider()

    # 시스템 프롬프트를 사용자가 직접 수정 가능하게
    st.subheader("📝 시스템 프롬프트")
    system_text = st.text_area(
        label="시스템 프롬프트",
        value="당신은 대학교의 친절한 AI 튜터입니다. 사용자의 질문에 반드시 '한국어(Korean)'로 친절하고 상세하게 답변해 주세요.",
        height=120,
        label_visibility="collapsed",
    )

    st.divider()

    # 대화 히스토리 길이 제한 설정
    st.subheader("🗂️ 히스토리 설정")
    max_history = st.slider("최근 대화 유지 수 (턴)", min_value=4, max_value=40, value=20, step=2)

    st.divider()

    col1, col2 = st.columns(2)

    # 대화 초기화
    with col1:
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.conversation = []
            st.session_state.messages = []
            st.rerun()

    # 대화 내보내기 기능
    with col2:
        if st.session_state.get("messages"):
            conversation_json = json.dumps(
                st.session_state.messages, ensure_ascii=False, indent=2
            )
            st.download_button(
                label="💾 대화 저장",
                data=conversation_json,
                file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.button("💾 대화 저장", disabled=True, use_container_width=True)

    st.divider()
    st.caption(f"현재 대화: {len(st.session_state.get('messages', []))}턴")


st.title("🧠 멀티 모델 지원 Bedrock 에이전트")
st.caption(f"🚀 현재 장착된 두뇌: **{selected_model}**")

# ==========================================
# 4. 세션 상태 초기화
# ==========================================
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# 전체 대화 내역 출력 (인덱스 + 사용 도구 뱃지 포함)
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        # 사용한 도구 뱃지 표시 (assistant 메시지에만)
        used_tools = msg.get("used_tools", [])
        if used_tools:
            badge_cols = st.columns(len(used_tools))
            for col, badge in zip(badge_cols, used_tools):
                col.markdown(badge)
        st.write(f"`[{i}]` {msg['content']}")


# ==========================================
# 5. AWS 클라이언트 캐싱 (최적화)
# ==========================================
@st.cache_resource
def get_aws_clients():
    my_config = Config(read_timeout=300, retries={"max_attempts": 3})
    b_client = boto3.client("bedrock-runtime", region_name="us-east-1", config=my_config)
    l_client = boto3.client("lambda", region_name="us-east-1")
    return b_client, l_client

bedrock_client, lambda_client = get_aws_clients()


# ==========================================
# 6. 스트림 파싱 함수
# ==========================================
def parse_stream_and_update_ui(response_stream, placeholder, prefix=""):
    ai_generated_text = ""
    tool_uses = []
    current_tool = None

    if prefix:
        placeholder.markdown(prefix + "▌")

    for event in response_stream["stream"]:
        if "contentBlockStart" in event:
            start = event["contentBlockStart"]["start"]
            if "toolUse" in start:
                current_tool = {
                    "toolUseId": start["toolUse"]["toolUseId"],
                    "name": start["toolUse"]["name"],
                    "input": "",
                }

        elif "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                ai_generated_text += delta["text"]
                placeholder.markdown(prefix + ai_generated_text + "▌")
            elif "toolUse" in delta:
                current_tool["input"] += delta["toolUse"]["input"]

        elif "contentBlockStop" in event:
            if current_tool:
                try:
                    current_tool["input"] = json.loads(current_tool["input"])
                except json.JSONDecodeError:
                    current_tool["input"] = {}
                tool_uses.append(current_tool)
                current_tool = None

    placeholder.markdown(prefix + ai_generated_text)
    return ai_generated_text, tool_uses


# ==========================================
# 7. 메인 챗봇 로직 (Agentic Loop)
# ==========================================
if user_input := st.chat_input("질문 (예: 뉴욕 시간이랑 런던 날씨 알려줘)"):

    # 사용자 입력 저장 및 출력
    turn_index = len(st.session_state.messages)
    st.chat_message("user").write(f"`[{turn_index}]` {user_input}")
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.conversation.append({"role": "user", "content": [{"text": user_input}]})

    with st.chat_message("assistant"):
        badge_placeholder = st.empty()   # 도구 뱃지를 말풍선 상단에 표시
        message_placeholder = st.empty()
        used_tools = []   # 사용한 도구 뱃지 목록
        accumulated_ai_text = ""
        MAX_TURNS = 5

        for turn in range(MAX_TURNS):

            # 최근 N턴 히스토리만 전달 (토큰 비용 절감)
            recent_conversation = st.session_state.conversation[-max_history:]

            # LLM 질의 (스트리밍)
            try:
                response = bedrock_client.converse_stream(
                    modelId=selected_model,
                    system=[{"text": system_text}],
                    messages=recent_conversation,
                    toolConfig=tool_config,
                )
            except NoCredentialsError:
                st.error("❌ AWS 인증 정보가 없습니다. IAM 권한을 확인하세요.")
                break
            except ClientError as e:
                st.error(f"❌ Bedrock 호출 실패: {e.response['Error']['Code']} - {e.response['Error']['Message']}")
                break

            # 스트리밍 출력용 prefix (도구 뱃지는 badge_placeholder에 별도 표시)
            current_prefix = ""
            if accumulated_ai_text:
                current_prefix = accumulated_ai_text + "\n\n"

            ai_text, ai_tools = parse_stream_and_update_ui(response, message_placeholder, prefix=current_prefix)

            # LLM 응답을 대화 히스토리에 추가
            assistant_content = []
            if ai_text:
                assistant_content.append({"text": ai_text})
                accumulated_ai_text += ai_text
            for tool in ai_tools:
                assistant_content.append({"toolUse": tool})
            if assistant_content:
                st.session_state.conversation.append({"role": "assistant", "content": assistant_content})

            # 도구 요청이 없으면 루프 종료 (최종 답변)
            if not ai_tools:
                # 뱃지가 있으면 말풍선 상단에 표시
                if used_tools:
                    badge_placeholder.markdown("　".join(used_tools))
                assistant_index = len(st.session_state.messages)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": accumulated_ai_text,
                    "used_tools": used_tools,   # 사용 도구 목록 저장
                })
                message_placeholder.markdown(f"`[{assistant_index}]` {accumulated_ai_text}")
                break

            # 도구 실행
            tool_result_blocks = []
            with st.status(f"⚙️ {selected_model} 모델이 도구를 실행 중... (Turn {turn + 1})", expanded=True) as status:
                for tool in ai_tools:
                    tool_name    = tool["name"]
                    tool_use_id  = tool["toolUseId"]
                    tool_input   = tool["input"]

                    # AI 환각 방어 - 기본값 먼저 설정
                    result = json.dumps({
                        "status": "error",
                        "message": f"'{tool_name}'은(는) 사용할 수 없는 도구입니다."
                    }, ensure_ascii=False)

                    st.write(f"- 🔍 **요청:** `{tool_name}` ({tool_input})")

                    # --- 로컬 파이썬 Tool ---
                    if tool_name == "get_current_time":
                        badge = "✅ `[로컬 Python Tool]` `get_current_time`"
                        if badge not in used_tools:
                            used_tools.append(badge)
                            badge_placeholder.markdown("　".join(used_tools))
                        result = get_current_time(tool_input.get("timezone", "Asia/Seoul"))
                        st.write("  [Tool] ✅ 완료")

                    # --- AWS Lambda Tool ---
                    elif tool_name == "invoke_weather_lambda":
                        badge = "☁️ `[AWS Lambda]` `GetWeatherFunction`"
                        if badge not in used_tools:
                            used_tools.append(badge)
                            badge_placeholder.markdown("　".join(used_tools))
                        target_city = tool_input.get("city", "Seoul")

                        try:
                            payload_data = json.dumps({"city": target_city})
                            lambda_response = lambda_client.invoke(
                                FunctionName="GetWeatherFunction",
                                InvocationType="RequestResponse",
                                Payload=payload_data,
                            )
                            response_payload = json.loads(lambda_response["Payload"].read().decode("utf-8"))
                            weather_data = response_payload.get("body", response_payload) if isinstance(response_payload, dict) else response_payload
                            result = json.dumps({"status": "success", "data": weather_data}, ensure_ascii=False)
                            st.write("  [Lambda] ✅ 실제 호출 성공")

                        except NoCredentialsError:
                            st.warning("  [Lambda] ⚠️ AWS 인증 오류. 가상 데이터 반환")
                            result = json.dumps({"status": "mock", "data": "20°C 쾌청 (인증 오류로 인한 가상 데이터)"}, ensure_ascii=False)
                        except ClientError as e:
                            error_code = e.response["Error"]["Code"]
                            st.warning(f"  [Lambda] ⚠️ 호출 실패 ({error_code}). 가상 데이터 반환")
                            mock_data = {"Seoul": "22°C 맑음", "New_York": "15°C 비", "London": "12°C 흐림"}
                            result = json.dumps({"status": "mock", "data": mock_data.get(target_city, "20°C 쾌청")}, ensure_ascii=False)
                        except Exception as e:
                            st.warning(f"  [Lambda] ⚠️ 예상치 못한 오류: {e}. 가상 데이터 반환")
                            mock_data = {"Seoul": "22°C 맑음", "New_York": "15°C 비", "London": "12°C 흐림"}
                            result = json.dumps({"status": "mock", "data": mock_data.get(target_city, "20°C 쾌청")}, ensure_ascii=False)

                    # 개선: Tool 결과를 JSON 구조로 전달
                    tool_result_blocks.append({
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content": [{"text": result}],
                        }
                    })

                st.session_state.conversation.append({"role": "user", "content": tool_result_blocks})
                status.update(label=f"Turn {turn + 1} 완료! 모델에게 다시 묻습니다...", state="complete", expanded=False)

        else:
            # MAX_TURNS 도달 시 부분 결과 포함하여 저장
            partial_msg = f"⚠️ 최대 반복 횟수({MAX_TURNS}회) 초과\n\n**지금까지의 결과:**\n{accumulated_ai_text}"
            if used_tools:
                badge_placeholder.markdown("　".join(used_tools))
            st.warning(partial_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": partial_msg,
                "used_tools": used_tools,
            })
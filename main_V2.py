# main_final.py (정보 흐름 최적화, 최종 완성본)

# --- 1. 기본 라이브러리 import ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import json
from typing import List, TypedDict
import logging
from fastapi.middleware.cors import CORSMiddleware

# --- 2. 로깅 기본 설정 ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s:     %(message)s')

# --- 3. LangChain, LangGraph 및 AI 관련 라이브러리 import ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
import google.generativeai as genai
from langchain_core.documents import Document

# --- 4. 환경 설정 ---
load_dotenv()

app = FastAPI(
    title="MOIT AI Agent Server v3.1",
    description="정보 흐름이 최적화된 최종 감독관 시스템",
    version="3.1.0",
)

# --- CORS 미들웨어 추가 ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI 모델 및 API 키 설정 ---
try:
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        logging.warning("GOOGLE_API_KEY가 .env 파일에 설정되지 않았습니다. 사진 분석 기능이 작동하지 않을 수 있습니다.")
    else:
        genai.configure(api_key=gemini_api_key)
except Exception as e:
    logging.warning(f"Gemini API 키 설정 실패: {e}")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
llm_for_meeting = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# --- 5. [복원] '모임 매칭 전문가(Self-RAG SubGraph)' 전체 정의 ---

# 5-1. 모임 매칭 전문가의 State 정의
class MeetingMatchingState(TypedDict):
    title: str
    description: str
    time: str
    location: str
    query: str
    context: List[Document]
    answer: str
    is_helpful: str
    rewrite_count: int

# 5-2. 모임 매칭 전문가의 각 노드(기능)들 정의
def prepare_query_node(state: MeetingMatchingState):
    logging.info("--- (모임 매칭) 1. 검색어 생성 노드 ---")
    prompt = ChatPromptTemplate.from_template(
        "당신은 사용자가 만들려는 모임의 상세 정보를 바탕으로, PineconeDB에서 유사 모임을 찾기 위한 최적의 검색어를 생성하는 AI입니다. "
        "아래 정보를 조합하여, 가장 핵심적인 키워드가 담긴 자연스러운 문장 형태의 검색어를 만들어주세요.\n\n"
        "[모임 정보]\n"
        "제목: {title}\n"
        "설명: {description}\n"
        "시간: {time}\n"
        "장소: {location}\n\n"
        "[작성 가이드]\n"
        "- '제목'과 '설명'에 담긴 핵심 활동이나 주제를 가장 중요한 키워드로 삼으세요.\n"
        "- '장소'는 중요한 참고 정보이지만, 너무 구체적인 장소 이름보다는 더 넓은 지역(예: '서울', '강남')을 포함하는 것이 좋습니다.\n"
        "- '시간' 정보는 검색어에 포함하지 않아도 좋습니다."
    )
    chain = prompt | llm_for_meeting | StrOutputParser()
    better_query = chain.invoke({
        "title": state.get("title", ""),
        "description": state.get("description", ""),
        "time": state.get("time", ""),
        "location": state.get("location", "")
    })
    # 초기 상태를 설정합니다.
    return {
        "title": state.get("title", ""),
        "description": state.get("description", ""),
        "time": state.get("time", ""),
        "location": state.get("location", ""),
        "query": better_query,
        "rewrite_count": 0
    }

def retrieve_node(state: MeetingMatchingState):
    logging.info(f"--- (모임 매칭) 2. 검색 노드 ({state.get('rewrite_count', 0)+1}번째) ---")
    meeting_index_name = os.getenv("PINECONE_INDEX_NAME_MEETING")
    embedding_function = OpenAIEmbeddings(model='text-embedding-3-large')
    vector_store = PineconeVectorStore.from_existing_index(index_name=meeting_index_name, embedding=embedding_function)
  
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={'score_threshold': 0.7, 'k': 2} 
    )
    context = retriever.invoke(state["query"])
    return {"context": context}


def generate_node(state: MeetingMatchingState):
    logging.info("--- (모임 매칭) 3. 답변 생성 노드 ---")
    context = state["context"]
    original_query = f"제목: {state['title']}, 설명: {state['description']}"
    context_str = "\n".join([f"모임 ID: {doc.metadata.get('meeting_id', 'N/A')}, 제목: {doc.metadata.get('title', 'N/A')}, 내용: {doc.page_content}" for doc in context])
    if not context:
        context_str = "유사한 모임을 찾지 못했습니다."
    
    prompt_str = """당신은 사용자의 요청을 매우 엄격하게 분석하여 유사한 모임을 추천하는 MOIT 플랫폼의 AI입니다.

[검색된 유사 모임 정보]
{context}

[사용자가 만들려는 모임 정보]
{query}

[지시사항]
1. [검색된 유사 모임 정보]가 "유사한 모임을 찾지 못했습니다."가 아닌 경우에만 아래 작업을 수행하세요.
2. [검색된 유사 모임 정보]와 [사용자가 만들려는 모임 정보]를 비교하여, 정말로 유사하다고 판단되는 모임만 골라주세요.
3. 사용자가 혹할 만한 매력적인 추천 문구를 작성해주세요.
4. 최종 답변은 반드시 아래와 같은 JSON 형식으로만 반환해야 합니다. 추가적인 설명은 절대 붙이지 마세요.

당신의 전체 응답은 다른 어떤 텍스트도 없이, 오직 '{{'로 시작해서 '}}'로 끝나는 유효한 JSON 객체여야 합니다.

[JSON 형식]
{{
    "summary": "요약 추천 문구 (예: '이런 모임은 어떠세요? 비슷한 주제의 모임이 이미 활발하게 활동 중이에요!')",
    "recommendations": [
        {{
            "meeting_id": "추천 모임의 ID",
            "title": "추천 모임의 제목"
        }}
    ]
}}

[검색된 모임이 없는 경우 JSON 형식]
{{
    "summary": "",
    "recommendations": []
}}
"""
    prompt = ChatPromptTemplate.from_template(prompt_str)
    chain = prompt | llm_for_meeting | StrOutputParser()
    answer = chain.invoke({"context": context_str, "query": original_query})
    return {"answer": answer}

def check_helpfulness_node(state: MeetingMatchingState):
    logging.info("--- (모임 매칭) 4. 유용성 검증 노드 ---")
    prompt = ChatPromptTemplate.from_template("당신은 AI가 생성한 추천이 사용자에게 정말 도움이 되는지 판단하는 검증 AI입니다. 'helpful' 또는 'unhelpful' 둘 중 하나로만 답변해주세요.\n\n[AI의 추천 내용]\n{answer}\n\n[사용자의 원래 요청]\n제목: {title}\n설명: {description}\n\n[검증 기준]\n- [AI 답변]의 `recommendations` 배열이 비어있지 않은지 확인하세요.\n- [AI 답변]의 `summary`가 긍정적인 추천 문구인지 확인하세요. (예: '비슷한 모임이 있어요' 등)\n- 위 두 조건이 모두 충족되고, 추천된 모임의 주제가 [원본 질문]과 관련이 있다면 'helpful'입니다.\n- 그 외 모든 경우는 'unhelpful'입니다.")
    chain = prompt | llm_for_meeting | StrOutputParser()
    # 파싱 오류를 방지하기 위해 strip()과 lower()를 추가합니다.
    raw_helpful = chain.invoke({"answer": state["answer"], "title": state["title"], "description": state["description"]})
    is_helpful = "helpful" if "helpful" in raw_helpful.strip().lower() else "unhelpful"
    return {"is_helpful": is_helpful}

def rewrite_query_node(state: MeetingMatchingState):
    logging.info("--- (모임 매칭) 5. 검색어 재작성 노드 ---")
    prompt = ChatPromptTemplate.from_template("당신은 이전 검색 결과가 만족스럽지 않아 검색어를 재작성하는 AI입니다. 사용자의 원래 의도를 바탕으로, 이전과는 다른 관점의 새로운 검색어를 제안해주세요.\n\n[이전 검색어]\n{query}")
    chain = prompt | llm_for_meeting | StrOutputParser()
    new_query = chain.invoke({"query": state["query"]})
    return {"query": new_query, "rewrite_count": state["rewrite_count"] + 1}

def decide_to_continue(state: MeetingMatchingState):
    return "end" if state["rewrite_count"] > 1 or state["is_helpful"] == "helpful" else "continue"

# 5-3. 모임 매칭 전문가 그래프(SubGraph) 조립
builder = StateGraph(MeetingMatchingState)
builder.add_node("prepare_query", prepare_query_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)
builder.add_node("check_helpfulness", check_helpfulness_node)
builder.add_node("rewrite_query", rewrite_query_node)
builder.set_entry_point("prepare_query")
builder.add_edge("prepare_query", "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "check_helpfulness")
builder.add_conditional_edges("check_helpfulness", decide_to_continue, {"continue": "rewrite_query", "end": END})
builder.add_edge("rewrite_query", "retrieve")
meeting_matching_agent = builder.compile()


# --- '모임 매칭 전문가'를 Tool로 포장하는 부분 ---
@tool
def run_meeting_matching_agent_tool(meeting_info: dict) -> str:
    """
    사용자가 만들려는 모임의 상세 정보(딕셔셔너리)를 입력받아, Self-RAG 기반의 지능형 에이전트를 실행하여 유사 모임을 찾아 추천 결과를 JSON 문자열로 반환합니다.
    """
    logging.info(f"--- 🤖 'Self-RAG 모임 매칭 전문가'를 호출합니다. 입력: {meeting_info} ---")
    try:
        # [★수정된 핵심 부분★]
        # 에이전트가 모든 루프를 마치고 내놓은 최종 답변을 그대로 신뢰하고 반환합니다.
        # 불필요한 'is_helpful' 외부 검증 로직을 제거합니다.
        final_state = meeting_matching_agent.invoke(meeting_info, {"recursion_limit": 5})
        logging.info("--- ✅ Self-RAG 모임 매칭이 성공적으로 완료되었습니다. ---")
        
        # 에이전트의 최종 답변을 그대로 반환
        return final_state.get("answer", json.dumps({"summary": "오류: 최종 답변을 생성하지 못했습니다.", "recommendations": []}))
            
    except Exception as e:
        logging.error(f"Self-RAG 모임 매칭 에이전트 실행 중 오류 발생: {e}", exc_info=True)
        return json.dumps({"summary": "모임 추천 중 심각한 오류가 발생했습니다.", "recommendations": []})


# 전문가 2: 사진 분석 전문가
@tool
def analyze_photo_tool(image_paths: list[str]) -> str:
    """
    사용자의 사진(이미지 파일 경로 리스트)을 입력받아, 그 사람의 성향, 분위기, 잠재적 관심사에 대한 텍스트 분석 결과를 반환합니다.
    """
    from PIL import Image
    try:
        logging.info(f"--- 📸 '사진 분석 전문가'가 작업을 시작합니다. (이미지 {len(image_paths)}개) ---")
        model = genai.GenerativeModel('gemini-2.5-flash')
        photo_analysis_prompt_text = "당신은 사람들의 일상 사진을 보고, 그 사람의 잠재적인 관심사와 성향을 추측하는 심리 분석가입니다. [분석할 사진] 아래 제공된 사진들 [지시사항] 1. 사진들 속 인물, 사물, 배경, 분위기를 종합적으로 분석하세요. 2. 사진 분석 결과를 바탕으로, 이 사람의 성향과 잠재적인 관심사를 3~4개의 핵심 키워드와 함께 설명해주세요. 3. 최종 결과는 다른 AI가 이해하기 쉽도록 간결한 분석 보고서 형식으로 작성해주세요."
        image_parts = [Image.open(path) for path in image_paths]
        response = model.generate_content([photo_analysis_prompt_text] + image_parts)
        logging.info("--- ✅ 사진 분석이 성공적으로 완료되었습니다. ---")
        return response.text
    except Exception as e:
        logging.error(f"사진 분석 중 오류 발생: {e}", exc_info=True)
        return f"오류: 사진 분석 중 문제가 발생했습니다: {e}"

# 전문가 3: 설문 분석 전문가
def _normalize(value, min_val, max_val):
    if value is None: return None
    return round((value - min_val) / (max_val - min_val), 4)

@tool
def analyze_survey_tool(survey_json_string: str) -> dict:
    """
    사용자의 설문 응답(JSON 문자열)을 입력받아, 수치적으로 정규화된 성향 프로필(딕셔너리)을 반환합니다.
    """
    logging.info("--- 📊 '설문 분석 전문가'가 작업을 시작합니다. ---")
    try:
        responses = json.loads(survey_json_string)
        features = {'FSC': {}, 'PSSR': {}, 'MP': {}, 'DLS': {}}
        features['FSC']['time_availability'] = _normalize(responses.get('1'), 1, 4)
        features['FSC']['financial_budget'] = _normalize(responses.get('2'), 1, 4)
        features['FSC']['energy_level'] = _normalize(responses.get('3'), 1, 5)
        features['FSC']['mobility'] = _normalize(responses.get('4'), 1, 5)
        features['FSC']['has_physical_constraints'] = True if responses.get('5') in [1, 2, 3] else False
        features['FSC']['has_housing_constraints'] = True if responses.get('12') in [2, 3, 4] else False
        features['FSC']['preferred_space'] = 'indoor' if responses.get('6') == 1 else 'outdoor'
        q13 = responses.get('13', 3); q14_r = 6 - responses.get('14', 3); q16 = responses.get('16', 3)
        self_criticism_raw = (q13 + q14_r + q16) / 3
        features['PSSR']['self_criticism_score'] = _normalize(self_criticism_raw, 1, 5)
        q15 = responses.get('15', 3); q18 = responses.get('18', 3); q20 = responses.get('20', 3)
        social_anxiety_raw = (q15 + q18 + q20) / 3
        features['PSSR']['social_anxiety_score'] = _normalize(social_anxiety_raw, 1, 5)
        features['PSSR']['isolation_level'] = _normalize(responses.get('21'), 1, 5)
        features['PSSR']['structure_preference_score'] = _normalize(responses.get('27'), 1, 5)
        features['PSSR']['avoidant_coping_score'] = _normalize(responses.get('29'), 1, 5)
        motivation_map = {1: 'achievement', 2: 'recovery', 3: 'connection', 4: 'vitality'}
        features['MP']['core_motivation'] = motivation_map.get(responses.get('31'))
        features['MP']['value_profile'] = {'knowledge': _normalize(responses.get('33'), 1, 5),'stability': _normalize(responses.get('34'), 1, 5),'relationship': _normalize(responses.get('35'), 1, 5),'health': _normalize(responses.get('36'), 1, 5),'creativity': _normalize(responses.get('37'), 1, 5),'control': _normalize(responses.get('38'), 1, 5),}
        features['MP']['process_orientation_score'] = _normalize(6 - responses.get('41', 3), 1, 5)
        sociality_map = {1: 'solo', 2: 'parallel', 3: 'low_interaction_group', 4: 'high_interaction_group'}
        features['DLS']['preferred_sociality_type'] = sociality_map.get(responses.get('39'))
        group_size_map = {1: 'one_on_one', 2: 'small_group', 3: 'large_group'}
        features['DLS']['preferred_group_size'] = group_size_map.get(responses.get('40'))
        features['DLS']['autonomy_preference_score'] = _normalize(responses.get('42'), 1, 5)
        logging.info("--- ✅ 설문 분석이 성공적으로 완료되었습니다. ---")
        return features
    except Exception as e:
        logging.error(f"설문 분석 중 오류 발생: {e}", exc_info=True)
        return {"error": f"설문 분석 중 오류가 발생했습니다: {e}"}

# 전문가 4: 설문 요약 전문가
@tool
def summarize_survey_profile_tool(survey_profile: dict) -> str:
    """
    'analyze_survey_tool'로부터 받은 정량적인 사용자 프로필(딕셔너리)을 입력받아,
    사람이 이해하기 쉬운 자연스러운 문장의 텍스트 요약 보고서로 변환합니다.
    """
    logging.info("--- ✍️ '설문 요약 전문가'가 작업을 시작합니다. ---")
    try:
        summarizer_prompt = ChatPromptTemplate.from_template("당신은 사용자의 성향 분석 데이터를 해석하여, 핵심적인 특징을 요약하는 프로파일러입니다. 아래 <사용자 프로필 데이터>를 보고, 이 사람의 성향을 한두 문단의 자연스러운 문장으로 요약해주세요.\n<사용자 프로필 데이터>\n{profile}\n[데이터 항목 설명] - FSC: 현실적인 제약 조건 (0에 가까울수록 제약이 큼), PSSR: 심리적 상태 (0에 가까울수록 안정적), MP: 활동 동기, DLS: 선호하는 사회성\n[요약 예시] '이 사용자는 현재 시간과 예산, 에너지 등 현실적인 제약이 크며, 사회적 불안감이 높아 혼자만의 활동을 통해 회복과 안정을 얻고 싶어하는 성향이 강하게 나타납니다.' 와 같이 간결하게 작성해주세요.")
        summarizer_chain = summarizer_prompt | llm | StrOutputParser()
        summary = summarizer_chain.invoke({"profile": survey_profile})
        logging.info("--- ✅ 설문 요약이 성공적으로 완료되었습니다. ---")
        return summary
    except Exception as e:
        logging.error(f"설문 요약 중 오류 발생: {e}", exc_info=True)
        return f"오류: 설문 요약 중 문제가 발생했습니다: {e}"


# --- 7. 감독관(Supervisor) 에이전트 생성 ---
tools = [
    run_meeting_matching_agent_tool,
    analyze_photo_tool,
    analyze_survey_tool,
    summarize_survey_profile_tool,
]

final_supervisor_prompt_v3_1 = """
당신은 사용자의 다양한 요청을 이해하고, 여러 AI 전문가들을 지휘하여 최적의 답변을 만들어내는 총괄 감독관입니다.

[당신이 지휘할 수 있는 전문가들]
- `run_meeting_matching_agent_tool`: 사용자가 만들려는 모임의 상세 정보(딕셔너리)를 받아 Self-RAG 방식으로 유사 모임을 찾아 추천합니다.
- `analyze_photo_tool`: 사진을 분석하여 사용자의 외면적 성향을 파악합니다.
- `analyze_survey_tool`: 설문 결과를 분석하여 사용자의 내면적 성향을 정량 데이터로 변환합니다.
- `summarize_survey_profile_tool`: 설문 분석 데이터를 사람이 이해하기 쉬운 텍스트로 요약합니다.

[작업 지침]
1.  사용자의 요청(`Human Message`)을 보고, 어떤 종류의 작업인지 명확히 파악하세요. ('유사 모임 추천' 또는 '새로운 취미 추천')
2.  작업에 필요한 전문가들을 호출하세요.
    - **('새로운 취미 추천'의 경우)** `analyze_survey_tool` -> `summarize_survey_profile_tool` -> `analyze_photo_tool` 순서로 호출하여 최종 답변을 생성합니다. 만약 사진이 제공되지 않으면, 사진 분석 단계를 건너뛰고 설문 분석 결과만으로 추천을 생성하세요.
    - **('유사 모임 추천'의 경우)** 사용자의 입력 데이터(딕셔너리 형태)를 `run_meeting_matching_agent_tool` 전문가의 `meeting_info` 인자로 그대로 전달하여 호출하고, 그 결과를 반환하면 됩니다.
3.  [매우 중요] 만약 '설문 요약'과 '사진 분석'의 결과가 서로 상반될 경우, 이 차이점을 명확히 인지하고 언급하며, 두 가지 성향을 모두 아우를 수 있는 균형 잡힌 최종 추천을 하는 것이 당신의 가장 중요한 임무입니다.
4.  모든 전문가의 보고를 종합하여, 사용자에게 전달할 최종 답변을 완성하세요.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", final_supervisor_prompt_v3_1),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

supervisor_agent = create_react_agent(llm, tools, prompt=prompt)


# --- 8. API 엔드포인트 정의 ---
class AgentInvokeRequest(BaseModel):
    messages: list

@app.post("/agent/invoke")
async def invoke_agent(request: AgentInvokeRequest):
    """
    사용자의 모든 요청을 받아 감독관 AI 에이전트를 실행하고 최종 답변을 반환합니다.
    """
    try:
        # 복잡한 딕셔너리 데이터를 json.dumps()를 사용해 하나의 문자열로 변환합니다.
        processed_messages = []
        for role, content in request.messages:
            if isinstance(content, dict):
                # content가 딕셔너리이면, JSON 문자열로 변환
                processed_messages.append((role, json.dumps(content, ensure_ascii=False)))
            else:
                # 딕셔너리가 아니면(일반 텍스트 등) 그대로 사용
                processed_messages.append((role, content))

        input_data = {"messages": processed_messages}

        final_answer = ""
        for event in supervisor_agent.stream(input_data, {"recursion_limit": 15}):
            if "messages" in event:
                last_message = event["messages"][-1]
                if isinstance(last_message.content, str) and not last_message.tool_calls:
                    final_answer = last_message.content
        
        return {"final_answer": final_answer}
    except Exception as e:
        logging.error(f"Agent 실행 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI 에이전트 처리 중 내부 서버 오류가 발생했습니다: {str(e)}")

# --- 9. Pinecone DB 업데이트/삭제 엔드포인트 ---
class NewMeeting(BaseModel):
    meeting_id: str
    title: str
    description: str
    time: str
    location: str

@app.post("/meetings/add")
async def add_meeting_to_pinecone(meeting: NewMeeting):
    try:
        meeting_index_name = os.getenv("PINECONE_INDEX_NAME_MEETING")
        if not meeting_index_name: raise ValueError("'.env' 파일에 PINECONE_INDEX_NAME_MEETING이(가) 설정되지 않았습니다.")
        
        embedding_function = OpenAIEmbeddings(model='text-embedding-3-large')
        vector_store = PineconeVectorStore.from_existing_index(index_name=meeting_index_name, embedding=embedding_function)
        
        full_text = f"제목: {meeting.title}\n설명: {meeting.description}\n시간: {meeting.time}\n장소: {meeting.location}"
        metadata = {"title": meeting.title, "description": meeting.description, "time": meeting.time, "location": meeting.location, "meeting_id": meeting.meeting_id}
        
        vector_store.add_texts(texts=[full_text], metadatas=[metadata], ids=[meeting.meeting_id])
        
        logging.info(f"--- Pinecone에 모임 추가 성공 (ID: {meeting.meeting_id}) ---")
        return {"status": "success", "message": f"모임(ID: {meeting.meeting_id})이 성공적으로 추가되었습니다."}
    except Exception as e:
        logging.error(f"Pinecone 업데이트 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pinecone에 모임을 추가하는 중 오류가 발생했습니다: {str(e)}")

@app.delete("/meetings/delete/{meeting_id}")
async def delete_meeting_from_pinecone(meeting_id: str):
    try:
        logging.info(f"--- Pinecone에서 모임 삭제 시작 (ID: {meeting_id}) ---")
        meeting_index_name = os.getenv("PINECONE_INDEX_NAME_MEETING")
        if not meeting_index_name: raise ValueError("'.env' 파일에 PINECONE_INDEX_NAME_MEETING이(가) 설정되지 않았습니다.")
        
        embedding_function = OpenAIEmbeddings(model='text-embedding-3-large')
        vector_store = PineconeVectorStore.from_existing_index(index_name=meeting_index_name, embedding=embedding_function)
        
        vector_store.delete(ids=[meeting_id])
        
        logging.info(f"--- Pinecone에서 모임 삭제 성공 (ID: {meeting_id}) ---")
        return {"status": "success", "message": f"모임(ID: {meeting_id})이 성공적으로 삭제되었습니다."}
    except Exception as e:
        logging.error(f"Pinecone 삭제 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pinecone에서 모임을 삭제하는 중 오류가 발생했습니다: {str(e)}")
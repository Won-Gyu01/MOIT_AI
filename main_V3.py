# main_final_v5.py (검증된 원본 기반 최종 이식 버전)

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

# --- 4. 환경 설정 및 FastAPI 앱 초기화 ---
load_dotenv()
app = FastAPI(title="MOIT AI Final Stable Server", version="5.0.0")

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
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
    else:
        logging.warning("GOOGLE_API_KEY가 .env 파일에 설정되지 않았습니다. 사진 분석 기능이 작동하지 않을 수 있습니다.")
except Exception as e:
    logging.warning(f"Gemini API 키 설정 실패: {e}")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
llm_for_meeting = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# --- 5. 마스터 에이전트 로직 전체 정의 ---

# 5-1. 마스터 에이전트의 State 정의
class MasterAgentState(TypedDict):
    user_input: dict
    route: str
    final_answer: str

# 5-2. 라우터 노드 정의 (안정성이 검증된 기존 방식 그대로 사용)
routing_prompt = ChatPromptTemplate.from_template(
    """당신은 사용자의 요청을 분석하여 어떤 담당자에게 전달해야 할지 결정하는 AI 라우터입니다.
    사용자의 요청을 보고, 아래 두 가지 경로 중 가장 적절한 경로 하나만 골라 그 이름만 정확히 답변해주세요.

    [경로 설명]
    1. `meeting_matching`: 사용자가 '새로운 모임'을 만들려고 할 때, 기존에 있던 '유사한 모임'을 추천해주는 경로입니다. (입력에 title, description 등이 포함됩니다)
    2. `hobby_recommendation`: 사용자에게 '새로운 취미' 자체를 추천해주는 경로입니다. (입력에 survey, hobby_info 등이 포함됩니다)

    [사용자 요청]:
    {user_input}

    [판단 결과 (meeting_matching 또는 hobby_recommendation)]:
    """
)
router_chain = routing_prompt | llm | StrOutputParser()

def route_request(state: MasterAgentState):
    """사용자의 입력을 보고 어떤 전문가에게 보낼지 결정하는 노드"""
    print("--- ROUTING ---")
    route_decision = router_chain.invoke({"user_input": state['user_input']})
    cleaned_decision = route_decision.strip().lower().replace("'", "").replace('"', '')
    print(f"라우팅 결정: {cleaned_decision}")
    return {"route": cleaned_decision}

# 5-3. 전문가 호출 노드들 정의

# 전문가 1: 모임 매칭 에이전트 (SubGraph) - 안정성이 검증된 기존 코드 전체를 그대로 사용
def call_meeting_matching_agent(state: MasterAgentState):
    """'모임 매칭 에이전트'를 독립적인 SubGraph로 실행하고 결과를 받아오는 노드"""
    print("--- CALLING: Meeting Matching Agent (Stable Version) ---")
    
    class MeetingAgentState(TypedDict):
        title: str; description: str; time: str; location: str; query: str;
        context: List[Document]; answer: str; rewrite_count: int; decision: str

    meeting_index_name = os.getenv("PINECONE_INDEX_NAME_MEETING")
    if not meeting_index_name: raise ValueError("'.env' 파일에 PINECONE_INDEX_NAME_MEETING 변수를 설정해야 합니다.")
    
    embedding_function = OpenAIEmbeddings(model='text-embedding-3-large')
    vector_store = PineconeVectorStore.from_existing_index(index_name=meeting_index_name, embedding=embedding_function)
    retriever = vector_store.as_retriever(search_kwargs={'k': 2})

    # SubGraph의 모든 노드와 프롬프트를 이 함수 내에서 직접 정의 (기존 방식과 동일)
    prepare_query_prompt = ChatPromptTemplate.from_template(
        "당신은 사용자가 입력한 정보를 바탕으로 유사한 다른 정보를 검색하기 위한 최적의 검색어를 만드는 전문가입니다.\n"
        "아래 [모임 정보]를 종합하여, 벡터 데이터베이스에서 유사한 모임을 찾기 위한 가장 핵심적인 검색 질문을 한 문장으로 만들어주세요.\n"
        "[모임 정보]:\n- 제목: {title}\n- 설명: {description}\n- 시간: {time}\n- 장소: {location}"
    )
    prepare_query_chain = prepare_query_prompt | llm_for_meeting | StrOutputParser()
    def prepare_query(m_state: MeetingAgentState):
        query = prepare_query_chain.invoke(m_state)
        return {"query": query, "rewrite_count": 0}

    def retrieve(m_state: MeetingAgentState):
        return {"context": retriever.invoke(m_state['query'])}

    generate_prompt = ChatPromptTemplate.from_template(
        "당신은 MOIT 플랫폼의 친절한 모임 추천 AI입니다. 사용자에게 \"혹시 이런 모임은 어떠세요?\" 라고 제안하는 말투로, "
        "반드시 아래 [검색된 정보]를 기반으로 유사한 모임이 있다는 것을 명확하게 설명해주세요.\n[검색된 정보]:\n{context}\n[사용자 질문]:\n{query}"
    )
    generate_chain = generate_prompt | llm_for_meeting | StrOutputParser()
    def generate(m_state: MeetingAgentState):
        context = "\n\n".join(doc.page_content for doc in m_state['context'])
        answer = generate_chain.invoke({"context": context, "query": m_state['query']})
        return {"answer": answer}

    check_helpfulness_prompt = ChatPromptTemplate.from_template(
        "당신은 AI 답변을 평가하는 엄격한 평가관입니다. 주어진 [AI 답변]이 사용자의 [원본 질문] 의도에 대해 유용한 제안을 하는지 평가해주세요. "
        "'helpful' 또는 'unhelpful' 둘 중 하나로만 답변해야 합니다.\n[원본 질문]: {query}\n[AI 답변]: {answer}"
    )
    check_helpfulness_chain = check_helpfulness_prompt | llm_for_meeting | StrOutputParser()
    def check_helpfulness(m_state: MeetingAgentState):
        result = check_helpfulness_chain.invoke(m_state)
        return {"decision": "helpful" if 'helpful' in result.lower() else "unhelpful"}

    rewrite_query_prompt = ChatPromptTemplate.from_template(
        "당신은 사용자의 질문을 더 좋은 검색 결과가 나올 수 있도록 명확하게 다듬는 프롬프트 엔지니어입니다. 주어진 [원본 질문]을 바탕으로, "
        "벡터 데이터베이스에서 더 관련성 높은 모임 정보를 찾을 수 있는 새로운 검색 질문을 하나만 만들어주세요.\n[원본 질문]: {query}"
    )
    rewrite_query_chain = rewrite_query_prompt | llm_for_meeting | StrOutputParser()
    def rewrite_query(m_state: MeetingAgentState):
        new_query = rewrite_query_chain.invoke(m_state)
        count = m_state.get('rewrite_count', 0) + 1
        return {"query": new_query, "rewrite_count": count}
    
    graph_builder = StateGraph(MeetingAgentState)
    graph_builder.add_node("prepare_query", prepare_query)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.add_node("check_helpfulness", check_helpfulness)
    graph_builder.add_node("rewrite_query", rewrite_query)
    graph_builder.set_entry_point("prepare_query")
    graph_builder.add_edge("prepare_query", "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", "check_helpfulness")
    graph_builder.add_conditional_edges( "check_helpfulness", lambda state: state['decision'], {"helpful": END, "unhelpful": "rewrite_query"})
    graph_builder.add_edge("rewrite_query", "retrieve")
    meeting_agent = graph_builder.compile()

    user_input = state['user_input'].get("meeting_info", state['user_input'])
    initial_state = { "title": user_input.get("title", ""), "description": user_input.get("description", ""), "time": user_input.get("time", ""), "location": user_input.get("location", "") }
    
    final_result_state = meeting_agent.invoke(initial_state, {"recursion_limit": 5})
    # 최종 결정이 unhelpful일 경우, 빈 추천을 반환하는 로직 추가
    if final_result_state.get("decision") != "helpful":
        return {"final_answer": json.dumps({"summary": "", "recommendations": []})}
    else:
        return {"final_answer": final_result_state.get("answer", "유사한 모임을 찾지 못했습니다.")}


# --- 7. [교체] 전문가 #2: 멀티모달 취미 추천 에이전트 (ReAct 감독관) ---

# 7-1. 취미 추천에 필요한 도구(Tool)들을 먼저 정의합니다.
@tool
def analyze_photo_tool(image_paths: list[str]) -> str:
    """사용자의 사진(이미지 파일 경로 리스트)을 입력받아, 그 사람의 성향, 분위기, 잠재적 관심사에 대한 텍스트 분석 결과를 반환합니다."""
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

def _normalize(value, min_val, max_val):
    if value is None: return None
    return round((value - min_val) / (max_val - min_val), 4)

@tool
def analyze_survey_tool(survey_json_string: str) -> dict:
    """사용자의 설문 응답(JSON 문자열)을 입력받아, 수치적으로 정규화된 성향 프로필(딕셔너리)을 반환합니다."""
    logging.info("--- 📊 '설문 분석 전문가'가 작업을 시작합니다. ---")
    try:
        responses = json.loads(survey_json_string)
        features = {'FSC': {}, 'PSSR': {}, 'MP': {}, 'DLS': {}}
        # (이하 전체 설문 분석 로직)
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

@tool
def summarize_survey_profile_tool(survey_profile: dict) -> str:
    """정량적인 사용자 프로필(딕셔너리)을 입력받아, 사람이 이해하기 쉬운 텍스트 요약 보고서로 변환합니다."""
    logging.info("--- ✍️ '설문 요약 전문가'가 작업을 시작합니다. ---")
    try:
        summarizer_prompt = ChatPromptTemplate.from_template("당신은 사용자의 성향 분석 데이터를 해석하여, 핵심적인 특징을 요약하는 프로파일러입니다. 아래 <사용자 프로필 데이터>를 보고, 이 사람의 성향을 한두 문단의 자연스러운 문장으로 요약해주세요.\n<사용자 프로필 데이터>\n{profile}\n[데이터 항목 설명] - FSC: 현실적인 제약 조건, PSSR: 심리적 상태, MP: 활동 동기, DLS: 선호하는 사회성\n[요약 예시] '이 사용자는 현재 시간과 예산, 에너지 등 현실적인 제약이 크며, 사회적 불안감이 높아 혼자만의 활동을 통해 회복과 안정을 얻고 싶어하는 성향이 강하게 나타납니다.' 와 같이 간결하게 작성해주세요.")
        summarizer_chain = summarizer_prompt | llm | StrOutputParser()
        summary = summarizer_chain.invoke({"profile": survey_profile})
        logging.info("--- ✅ 설문 요약이 성공적으로 완료되었습니다. ---")
        return summary
    except Exception as e:
        logging.error(f"설문 요약 중 오류 발생: {e}", exc_info=True)
        return f"오류: 설문 요약 중 문제가 발생했습니다: {e}"

# 7-2. 이 도구들을 지휘할 ReAct 감독관을 생성합니다.
hobby_tools = [analyze_photo_tool, analyze_survey_tool, summarize_survey_profile_tool]
hobby_supervisor_prompt = """당신은 사용자의 사진과 설문 결과를 종합하여 맞춤형 취미를 추천하는 AI 큐레이터입니다.
주어진 전문가들을 활용하여 다음 단계를 순서대로 수행하세요:
1. `analyze_survey_tool`로 설문을 분석합니다.
2. 그 결과를 `summarize_survey_profile_tool`로 요약합니다.
3. `analyze_photo_tool`로 사진을 분석합니다. (사진이 없다면 이 단계는 건너뜁니다.)
4. 두 결과를 종합하여 최종 추천 메시지를 생성합니다. 두 결과가 상반될 경우, 그 차이를 언급하며 균형잡힌 추천을 하는 것이 중요합니다.
"""
hobby_prompt = ChatPromptTemplate.from_messages([("system", hobby_supervisor_prompt), MessagesPlaceholder(variable_name="messages")])
hobby_supervisor_agent = create_react_agent(llm, hobby_tools, prompt=hobby_prompt)

# 7-3. 마스터 에이전트(라우터)가 호출할 최종 노드 함수를 만듭니다.
def call_multimodal_hobby_agent(state: MasterAgentState):
    """'멀티모달 취미 추천 감독관'을 호출하고 결과를 받아오는 노드"""
    print("--- CALLING: Multimodal Hobby Supervisor Agent ---")
    
    hobby_info = state["user_input"].get("hobby_info", state["user_input"])
    user_input_str = json.dumps(hobby_info, ensure_ascii=False)
    input_data = {"messages": [("user", f"다음 사용자 정보를 바탕으로 최종 취미 추천을 해주세요: {user_input_str}")]}
    
    final_answer = ""
    for event in hobby_supervisor_agent.stream(input_data, {"recursion_limit": 15}):
        if "messages" in event:
            last_message = event["messages"][-1]
            if isinstance(last_message.content, str) and not last_message.tool_calls:
                final_answer = last_message.content
                
    return {"final_answer": final_answer}


# --- 8. 마스터 에이전트(라우터) 조립 ---
master_graph_builder = StateGraph(MasterAgentState)

master_graph_builder.add_node("router", route_request)
master_graph_builder.add_node("meeting_matcher", call_meeting_matching_agent)
master_graph_builder.add_node("hobby_recommender", call_multimodal_hobby_agent) # [교체됨]

master_graph_builder.set_entry_point("router")

master_graph_builder.add_conditional_edges(
    "router", 
    lambda state: state['route'],
    {"meeting_matching": "meeting_matcher", "hobby_recommendation": "hobby_recommender"}
)

master_graph_builder.add_edge("meeting_matcher", END)
master_graph_builder.add_edge("hobby_recommender", END)

master_agent = master_graph_builder.compile()


# --- 9. API 엔드포인트 정의 ---
class UserRequest(BaseModel):
    user_input: dict

@app.post("/agent/invoke")
async def invoke_agent(request: UserRequest):
    try:
        input_data = {"user_input": request.user_input}
        result = master_agent.invoke(input_data)
        return {"final_answer": result.get("final_answer", "오류: 최종 답변을 생성하지 못했습니다.")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 에이전트 처리 중 내부 서버 오류가 발생했습니다: {e}")


# --- 10. Pinecone DB 업데이트/삭제 엔드포인트 ---
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
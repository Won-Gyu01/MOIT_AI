# main_final_v4.py (안정적인 뼈대 + 최신 엔진 이식 최종본)

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
from langgraph.prebuilt import create_react_agent # [새 엔진]을 위해 추가
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool # [새 엔진]을 위해 추가
import google.generativeai as genai # [새 엔진]을 위해 추가
from langchain_core.documents import Document

# --- 4. 환경 설정 ---
load_dotenv()

app = FastAPI(
    title="MOIT AI Final Hybrid Server",
    description="안정적인 라우터 기반에 최신 멀티모달 기능을 이식한 최종 AI 시스템",
    version="4.1.0",
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
        logging.warning("GOOGLE_API_KEY가 .env 파일에 설정되지 않았습니다.")
    else:
        genai.configure(api_key=gemini_api_key)
except Exception as e:
    logging.warning(f"Gemini API 키 설정 실패: {e}")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
llm_for_meeting = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# --- 5. 마스터 에이전트의 State(기억 상자) 정의 ---
class MasterAgentState(TypedDict):
    user_input: dict
    route: str
    final_answer: str


# --- 6. 전문가 #1: Self-RAG 모임 매칭 에이전트 (SubGraph) ---
# [유지] 안정성이 검증된 기존 모임 매칭 전문가 코드를 그대로 사용합니다.

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

    # SubGraph의 노드들을 정의합니다.
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
        "당신은 MOIT 플랫폼의 친절한 모임 추천 AI입니다... (이하 기존 프롬프트와 동일)" # 기존 generate_prompt 전체 삽입
    )
    generate_chain = generate_prompt | llm_for_meeting | StrOutputParser()
    def generate(m_state: MeetingAgentState):
        context = "\n\n".join(doc.page_content for doc in m_state['context'])
        answer = generate_chain.invoke({"context": context, "query": m_state['query']})
        return {"answer": answer}

    check_helpfulness_prompt = ChatPromptTemplate.from_template(
        "당신은 AI 답변을 평가하는 엄격한 평가관입니다... (이하 기존 프롬프트와 동일)" # 기존 check_helpfulness_prompt 전체 삽입
    )
    check_helpfulness_chain = check_helpfulness_prompt | llm_for_meeting | StrOutputParser()
    def check_helpfulness(m_state: MeetingAgentState):
        result = check_helpfulness_chain.invoke(m_state)
        return {"decision": "helpful" if 'helpful' in result.lower() else "unhelpful"}

    rewrite_query_prompt = ChatPromptTemplate.from_template(
        "당신은 사용자의 질문을 더 좋은 검색 결과가 나올 수 있도록... (이하 기존 프롬프트와 동일)" # 기존 rewrite_query_prompt 전체 삽입
    )
    rewrite_query_chain = rewrite_query_prompt | llm_for_meeting | StrOutputParser()
    def rewrite_query(m_state: MeetingAgentState):
        new_query = rewrite_query_chain.invoke(m_state)
        count = m_state.get('rewrite_count', 0) + 1
        return {"query": new_query, "rewrite_count": count}
    
    # SubGraph를 조립합니다.
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

    # 마스터 에이전트로부터 받은 정보로 SubGraph를 실행합니다.
    user_input = state['user_input']
    initial_state = { "title": user_input.get("title", ""), "description": user_input.get("description", ""), "time": user_input.get("time", ""), "location": user_input.get("location", "") }
    
    final_result_state = meeting_agent.invoke(initial_state, {"recursion_limit": 5})
    final_answer = final_result_state.get("answer", "유사한 모임을 찾지 못했습니다.")
    
    return {"final_answer": final_answer}


# --- 7. [교체] 전문가 #2: 멀티모달 취미 추천 에이전트 (ReAct 감독관) ---

# 7-1. 취미 추천에 필요한 도구(Tool)들을 정의합니다.
@tool
def analyze_photo_tool(image_paths: list[str]) -> str:
    from PIL import Image
    try:
        logging.info(f"--- 📸 '사진 분석 전문가'가 작업을 시작합니다. ---")
        model = genai.GenerativeModel('gemini-2.5-flash')
        # (이하 전체 사진 분석 프롬프트 및 로직)
        photo_analysis_prompt_text = "당신은 사람들의 일상 사진을 보고... (이하 생략)"
        image_parts = [Image.open(path) for path in image_paths]
        response = model.generate_content([photo_analysis_prompt_text] + image_parts)
        return response.text
    except Exception as e:
        return f"오류: 사진 분석 중 문제가 발생했습니다: {e}"

def _normalize(value, min_val, max_val):
    if value is None: return None
    return round((value - min_val) / (max_val - min_val), 4)

@tool
def analyze_survey_tool(survey_json_string: str) -> dict:
    logging.info("--- 📊 '설문 분석 전문가'가 작업을 시작합니다. ---")
    try:
        responses = json.loads(survey_json_string)
        features = {'FSC': {}, 'PSSR': {}, 'MP': {}, 'DLS': {}}
        # (이하 전체 설문 분석 로직)
        # ...
        return features
    except Exception as e:
        return {"error": f"설문 분석 중 오류가 발생했습니다: {e}"}

@tool
def summarize_survey_profile_tool(survey_profile: dict) -> str:
    logging.info("--- ✍️ '설문 요약 전문가'가 작업을 시작합니다. ---")
    try:
        summarizer_prompt = ChatPromptTemplate.from_template("당신은 사용자의 성향 분석 데이터를 해석하여... (이하 생략)")
        summarizer_chain = summarizer_prompt | llm | StrOutputParser()
        summary = summarizer_chain.invoke({"profile": survey_profile})
        return summary
    except Exception as e:
        return f"오류: 설문 요약 중 문제가 발생했습니다: {e}"

# 7-2. 이 도구들을 지휘할 ReAct 감독관을 생성합니다.
hobby_tools = [analyze_photo_tool, analyze_survey_tool, summarize_survey_profile_tool]
hobby_supervisor_prompt = """당신은 사용자의 사진과 설문 결과를 종합하여 맞춤형 취미를 추천하는 AI 큐레이터입니다... (이하 우리가 완성한 최종 감독관 프롬프트 v2.2)"""
hobby_prompt = ChatPromptTemplate.from_messages([("system", hobby_supervisor_prompt), MessagesPlaceholder(variable_name="messages")])
hobby_supervisor_agent = create_react_agent(llm, hobby_tools, prompt=hobby_prompt)

# 7-3. 마스터 에이전트(라우터)가 호출할 최종 노드 함수를 만듭니다.
def call_multimodal_hobby_agent(state: MasterAgentState):
    """'멀티모달 취미 추천 감독관'을 호출하고 결과를 받아오는 노드"""
    print("--- CALLING: Multimodal Hobby Supervisor Agent ---")
    
    user_input_str = json.dumps(state["user_input"], ensure_ascii=False)
    input_data = {"messages": [("user", f"다음 사용자 정보를 바탕으로 최종 취미 추천을 해주세요: {user_input_str}")]}
    
    final_answer = ""
    for event in hobby_supervisor_agent.stream(input_data, {"recursion_limit": 15}):
        if "messages" in event:
            last_message = event["messages"][-1]
            if isinstance(last_message.content, str) and not last_message.tool_calls:
                final_answer = last_message.content
                
    return {"final_answer": final_answer}


# --- 8. 마스터 에이전트(라우터) 조립 ---

# 8-1. 라우터 노드 정의
routing_prompt = ChatPromptTemplate.from_template(
    """당신은 사용자의 요청을 분석하여 어떤 담당자에게 전달해야 할지 결정하는 AI 라우터입니다... (이하 기존 라우터 프롬프트와 동일)"""
)
router_chain = routing_prompt | llm | StrOutputParser()

def route_request(state: MasterAgentState):
    print("--- ROUTING ---")
    # [수정] 라우팅을 위한 입력 데이터 구조를 명확히 합니다.
    task_description = state['user_input'].get('task_description', str(state['user_input']))
    route_decision = router_chain.invoke({"user_input": task_description})
    cleaned_decision = route_decision.strip().replace("'", "").replace('"', '')
    print(f"라우팅 결정: {cleaned_decision}")
    return {"route": cleaned_decision}

# 8-2. 마스터 그래프 조립
master_graph_builder = StateGraph(MasterAgentState)

master_graph_builder.add_node("router", route_request)
master_graph_builder.add_node("meeting_matcher", call_meeting_matching_agent)
# [교체됨]
master_graph_builder.add_node("hobby_recommender", call_multimodal_hobby_agent) 

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
# main.py (진짜 최종 완성본 - Self-RAG 루프 오류 수정 및 모든 내용 복원)

# --- 1. 기본 라이브러리 import ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests
import json
from typing import List, TypedDict, Optional
from fastapi.middleware.cors import CORSMiddleware
import logging # 로깅 라이브러리 추가

# --- 2. 로깅 기본 설정 ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s:     %(message)s')

# --- 3. LangChain 및 LangGraph 관련 라이브러리 import ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document


# --- 4. 환경 설정 ---
load_dotenv()

app = FastAPI(
    title="MOIT AI Agent Server",
    description="MOIT 플랫폼을 위한 멀티 에이전트 시스템 API",
    version="1.0.0",
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


# --- 5. 마스터 에이전트 로직 전체 정의 ---

# 5-1. 마스터 에이전트의 State(기억 상자) 정의
class MasterAgentState(TypedDict):
    user_input: dict
    route: str
    final_answer: str

# 5-2. 라우터 노드(지휘관의 두뇌) 정의
llm = ChatOpenAI(model="gpt-4o-mini")

routing_prompt = ChatPromptTemplate.from_template(
    """당신은 사용자의 요청을 분석하여 어떤 담당자에게 전달해야 할지 결정하는 AI 라우터입니다.
    사용자의 요청을 보고, 아래 두 가지 경로 중 가장 적절한 경로 하나만 골라 그 이름만 정확히 답변해주세요.

    [경로 설명]
    1. `meeting_matching`: 사용자가 '새로운 모임'을 만들려고 할 때, 기존에 있던 '유사한 모임'을 추천해주는 경로입니다. (입력에 title, description 등이 포함됩니다)
    2. `hobby_recommendation`: 사용자에게 '새로운 취미' 자체를 추천해주는 경로입니다. (입력에 survey, user_context 등이 포함됩니다)

    [사용자 요청]:
    {user_input}

    [판단 결과 (meeting_matching 또는 hobby_recommendation)]:
    """
)

router_chain = routing_prompt | llm | StrOutputParser()

def route_request(state: MasterAgentState):
    """사용자의 입력을 보고 어떤 전문가에게 보낼지 결정하는 노드"""
    logging.info("--- ROUTING ---")
    route_decision = router_chain.invoke({"user_input": state['user_input']})
    cleaned_decision = route_decision.strip().replace("'", "").replace('"', '')
    logging.info(f"라우팅 결정: {cleaned_decision}")
    return {"route": cleaned_decision}


# 5-3. 전문가 호출 노드들 정의

# 전문가 1: 모임 매칭 에이전트 (SubGraph)
def call_meeting_matching_agent(state: MasterAgentState):
    """'모임 매칭 에이전트'를 독립적인 SubGraph로 실행하고 결과를 받아오는 노드"""
    logging.info("--- CALLING: Meeting Matching Agent ---")
    
    class MeetingAgentState(TypedDict):
        title: str; description: str; time: str; location: str; query: str;
        context: List[Document]; answer: str; rewrite_count: int; decision: str

    meeting_llm = ChatOpenAI(model="gpt-4o-mini")
    meeting_index_name = os.getenv("PINECONE_INDEX_NAME_MEETING")
    if not meeting_index_name: raise ValueError("'.env' 파일에 PINECONE_INDEX_NAME_MEETING 변수를 설정해야 합니다.")
    
    embedding_function = OpenAIEmbeddings(model='text-embedding-3-large')
    vector_store = PineconeVectorStore.from_existing_index(index_name=meeting_index_name, embedding=embedding_function)
    # [수정!] 유사도 점수 임계값을 설정하여 관련성 높은 문서만 가져옵니다.
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.75, 'k': 2}
    )  

    # SubGraph의 노드들을 정의합니다. (프롬프트 복원)
    prepare_query_prompt = ChatPromptTemplate.from_template(
        """당신은 사용자가 입력한 정보를 바탕으로 유사한 다른 정보를 검색하기 위한 최적의 검색어를 만드는 전문가입니다.
        아래 [모임 정보]를 종합하여, 벡터 데이터베이스에서 유사한 모임을 찾기 위한 가장 핵심적인 검색 질문을 한 문장으로 만들어주세요.
        [모임 정보]:
- 제목: {title}
- 설명: {description}
- 시간: {time}
- 장소: {location}"""
    )
    prepare_query_chain = prepare_query_prompt | meeting_llm | StrOutputParser()
    def prepare_query(m_state: MeetingAgentState):
        logging.info("--- (Sub) Preparing Query ---")
        query = prepare_query_chain.invoke({"title": m_state['title'], "description": m_state['description'], "time": m_state.get('time', ''), "location": m_state.get('location', '')})
        logging.info(f"생성된 검색어: {query}")
        return {"query": query}

    def retrieve(m_state: MeetingAgentState):
        logging.info("--- (Sub) Retrieving Context from DB ---")
        context = retriever.invoke(m_state['query'])
        logging.info(f"DB에서 {len(context)}개의 유사 문서를 찾았습니다.")
        return {"context": context}

    # [최종 추천 프롬프트]
    generate_prompt = ChatPromptTemplate.from_template(
        """당신은 사용자의 요청을 매우 엄격하게 분석하여 유사한 모임을 추천하는 MOIT 플랫폼의 AI입니다.
        사용자가 만들려는 모임과 **주제, 활동 내용이 명확하게 일치하는** 기존 모임만 추천해야 합니다.

        [사용자 입력 정보]:
        {query}

        [검색된 유사 모임 정보]:
        {context}

        [지시사항]:
        1. [검색된 유사 모임 정보]의 각 항목을 [사용자 입력 정보]와 비교하여, **정말로 관련성이 높다고 판단되는 모임만** 골라냅니다. (예: '축구' 모임을 찾는 사용자에게 '야구' 모임은 추천하지 않습니다.)
        2. 1번에서 골라낸 모임이 있다면, 해당 모임을 기반으로 사용자에게 제안할 추천사를 친절한 말투("~는 어떠세요?")로 작성합니다.
        3. 1번에서 골라낸 모임의 `meeting_id`와 `title`을 추출하여 `recommendations` 배열을 구성합니다. **추천할 모임이 하나뿐이라면, 배열에 하나만 포함합니다.**
        4. 최종 답변을 아래와 같은 JSON 형식으로만 제공해주세요. 다른 텍스트는 절대 포함하지 마세요.

        [JSON 출력 형식 예시]:
        // 추천할 모임이 1개일 경우
        {{
            "summary": "비슷한 축구 모임이 있는데, 참여해 보시는 건 어떠세요?",
            "recommendations": [
                {{ "meeting_id": "축구 모임 ID", "title": "같이 축구하실 분!" }}
            ]
        }}

        // 1번에서 골라낸 모임이 없을 경우 (추천할 만한 모임이 없는 경우)
        {{
            "summary": "",
            "recommendations": []
        }}
        """
    )
    generate_chain = generate_prompt | meeting_llm | StrOutputParser()
    def generate(m_state: MeetingAgentState):
        logging.info("--- (Sub) Generating Final Answer ---")
        context_str = ""
        for i, doc in enumerate(m_state['context']):
            metadata = doc.metadata or {}
            meeting_id = metadata.get('meeting_id', 'N/A')
            title = metadata.get('title', 'N/A')
            context_str += f"모임 {i+1}:\n  - meeting_id: {meeting_id}\n  - title: {title}\n  - content: {doc.page_content}\n\n"
        if not m_state['context']: context_str = "유사한 모임을 찾지 못했습니다."
        answer = generate_chain.invoke({"context": context_str, "query": m_state['query']})
        return {"answer": answer}

    # [수정!] LLM이 한 단어로만 답하도록 프롬프트를 강화합니다.
    check_helpfulness_prompt = ChatPromptTemplate.from_template(
        """당신은 AI 답변을 평가하는 엄격한 평가관입니다. 주어진 [AI 답변]이 사용자의 [원본 질문] 의도에 대해 유용한 제안을 하는지 평가해주세요.
        다른 설명은 일절 추가하지 말고, 오직 'helpful' 또는 'unhelpful' 둘 중 하나의 단어로만 답변해야 합니다.

        [원본 질문]: {query}
        [AI 답변]: {answer}
        
        [평가 결과 (helpful 또는 unhelpful)]:"""
    )
    check_helpfulness_chain = check_helpfulness_prompt | meeting_llm | StrOutputParser()
    def check_helpfulness(m_state: MeetingAgentState):
        logging.info("--- (Sub) Checking Helpfulness ---")
        raw_result = check_helpfulness_chain.invoke({"query": m_state['query'], "answer": m_state['answer']})
        
        # [수정!] 더 안전한 파싱 로직과 명확한 로깅
        cleaned_result = raw_result.strip().lower().replace('"', '').replace("'", "")
        decision = "helpful" if cleaned_result == "helpful" else "unhelpful"
        
        logging.info(f"답변 유용성 평가 (Raw): {raw_result} -> (Parsed): {decision}")
        return {"decision": decision}

    rewrite_query_prompt = ChatPromptTemplate.from_template(
        """당신은 더 나은 검색 결과를 위해 질문을 재구성하는 프롬프트 엔지니어입니다.
        [원본 질문]은 벡터 검색에서 좋은 결과를 얻지 못했습니다. 원본 질문의 핵심 의도는 유지하되, 완전히 다른 관점에서 접근하거나, 더 구체적인 키워드를 사용하여 관련성 높은 모임을 찾을 수 있는 새로운 검색 질문을 하나만 만들어주세요.
        [원본 질문]: {query}
        [새로운 검색 질문]:"""
    )
    rewrite_query_chain = rewrite_query_prompt | meeting_llm | StrOutputParser()
    def rewrite_query(m_state: MeetingAgentState):
        logging.info("--- (Sub) Rewriting Query ---")
        new_query = rewrite_query_chain.invoke({"query": m_state['query']})
        logging.info(f"재작성된 검색어: {new_query}")
        count = m_state.get('rewrite_count', 0) + 1
        return {"query": new_query, "rewrite_count": count}
    
    def decide_to_continue(state: MeetingAgentState):
        if state.get("rewrite_count", 0) >= 2: 
            logging.info(f"--- 재시도 횟수({state.get('rewrite_count', 0)}) 초과했기에 루프를 종료합니다. ---")
            return END
        if state.get("decision") == "helpful":
            return END
        return "rewrite_query"
    
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
    graph_builder.add_conditional_edges("check_helpfulness", decide_to_continue)
    graph_builder.add_edge("rewrite_query", "retrieve")
    meeting_agent = graph_builder.compile()

    user_input = state['user_input']
    initial_state = {
        "title": user_input.get("title", ""), "description": user_input.get("description", ""),
        "time": user_input.get("time", ""), "location": user_input.get("location", ""),
        "rewrite_count": 0
    }
    
    final_result_state = meeting_agent.invoke(initial_state, {"recursion_limit": 15})

    final_decision = final_result_state.get("decision")
    final_answer = final_result_state.get("answer")

    if final_decision == 'helpful':
        logging.info("--- 최종 결정이 'helpful'이므로, 생성된 추천안을 반환합니다. ---")
        return {"final_answer": final_answer}
    else:
        logging.info("--- 최종 결정이 'helpful'이 아니므로, 신규 생성을 유도하기 위해 빈 추천안을 반환합니다. ---")
        empty_recommendation = json.dumps({"summary": "", "recommendations": []})
        return {"final_answer": empty_recommendation}


# 전문가 2: 취미 추천 에이전트 (Tool)
def call_hobby_recommendation_agent(state: MasterAgentState):
    """'취미 추천 에이전트(API)'를 호출하고 결과를 받아오는 노드"""
    logging.info("--- CALLING: Hobby Recommendation Agent ---")
    
    url = "http://127.0.0.1:5000/recommend"
    
    try:
        response = requests.post(url, json=state['user_input'])
        response.raise_for_status()
        recommendations = response.json()
        
        if not recommendations:
            final_answer = "아쉽지만 현재 조건에 맞는 취미를 찾지 못했어요."
        else:
            top3 = recommendations[:3]
            final_answer = json.dumps(top3, ensure_ascii=False)
            
    except requests.exceptions.RequestException as e:
        final_answer = "취미 추천 서버에 문제가 발생하여 연결할 수 없습니다."
        
    return {"final_answer": final_answer}


# 5-4. 마스터 에이전트 그래프 조립 및 컴파일
master_graph_builder = StateGraph(MasterAgentState)

master_graph_builder.add_node("router", route_request)
master_graph_builder.add_node("meeting_matcher", call_meeting_matching_agent)
master_graph_builder.add_node("hobby_recommender", call_hobby_recommendation_agent)

master_graph_builder.set_entry_point("router")

master_graph_builder.add_conditional_edges(
    "router", 
    lambda state: state['route'],
    {"meeting_matching": "meeting_matcher", "hobby_recommendation": "hobby_recommender"}
)

master_graph_builder.add_edge("meeting_matcher", END)
master_graph_builder.add_edge("hobby_recommender", END)

master_agent = master_graph_builder.compile()


# --- 6. API 엔드포인트(주소) 정의 ---

# 6-1. 메인 AI 요청 처리 엔드포인트
class UserRequest(BaseModel):
    user_input: dict

@app.post("/agent/invoke")
async def invoke_agent(request: UserRequest):
    try:
        input_data = {"user_input": request.user_input}
        result = master_agent.invoke(input_data, {"recursion_limit": 5}) # 마스터 에이전트에도 안전장치 추가
        # AI의 답변이 이미 JSON 문자열일 수 있으므로, 그대로 반환하거나 파싱해서 재구성할 수 있습니다.
        # 여기서는 백엔드와의 약속에 따라 그대로 반환하는 것이 안전합니다.
        return {"final_answer": result.get("final_answer", "오류: 최종 답변을 생성하지 못했습니다.")}
    except Exception as e:
        logging.error(f"Agent 실행 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="AI 에이전트 처리 중 내부 서버 오류가 발생했습니다.")


# 6-2. Pinecone DB 업데이트 엔드포인트
class NewMeeting(BaseModel):
    meeting_id: str
    title: str
    description: str
    time: str
    location: str

@app.post("/meetings/add")
async def add_meeting_to_pinecone(meeting: NewMeeting):
    try:
        logging.info(f"--- Pinecone에 새로운 모임 추가 시작 (ID: {meeting.meeting_id}) ---")
        meeting_index_name = os.getenv("PINECONE_INDEX_NAME_MEETING")
        if not meeting_index_name: raise ValueError("'.env' 파일에 PINECONE_INDEX_NAME_MEETING이(가) 설정되지 않았습니다.")
        
        embedding_function = OpenAIEmbeddings(model='text-embedding-3-large')
        vector_store = PineconeVectorStore.from_existing_index(index_name=meeting_index_name, embedding=embedding_function)
        
        full_text = f"제목: {meeting.title}\n설명: {meeting.description}\n시간: {meeting.time}\n장소: {meeting.location}"
        # [수정!] Pinecone 메타데이터에 meeting_id를 필수로 포함
        metadata = {
            "title": meeting.title, 
            "description": meeting.description, 
            "time": meeting.time, 
            "location": meeting.location,
            "meeting_id": meeting.meeting_id 
        }
        
        vector_store.add_texts(texts=[full_text], metadatas=[metadata], ids=[meeting.meeting_id])
        
        logging.info(f"--- Pinecone에 모임 추가 성공 (ID: {meeting.meeting_id}) ---")
        return {"status": "success", "message": f"모임(ID: {meeting.meeting_id})이 성공적으로 추가되었습니다."}
    except Exception as e:
        logging.error(f"Pinecone 업데이트 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pinecone에 모임을 추가하는 중 오류가 발생했습니다: {str(e)}")


# 6-3. Pinecone DB 삭제 엔드포인트
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
        raise HTTPException(status_code=500, detail=f"Pinecone에서 모임을 삭제하는 중 오류가 발생했습니다.: {str(e)}")
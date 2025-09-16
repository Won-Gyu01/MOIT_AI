# main.py (최종 완성본)

# --- 1. 기본 라이브러리 import ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests
import json
from typing import List, TypedDict, Optional

# --- 2. LangChain 및 LangGraph 관련 라이브러리 import ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

# --- 3. 환경 설정 ---
load_dotenv()
app = FastAPI(
    title="MOIT AI Agent Server",
    description="MOIT 플랫폼을 위한 멀티 에이전트 시스템 API",
    version="1.0.0",
)

# --- 4. 마스터 에이전트 로직 전체 정의 ---

# 4-1. 마스터 에이전트의 State 정의
class MasterAgentState(TypedDict):
    user_input: dict
    route: str
    final_answer: str

# 4-2. 라우터 노드 정의
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
    print("--- ROUTING ---")
    route_decision = router_chain.invoke({"user_input": state['user_input']})
    cleaned_decision = route_decision.strip().replace("'", "").replace('"', '')
    print(f"라우팅 결정: {cleaned_decision}")
    return {"route": cleaned_decision}

# 4-3. 전문가 호출 노드들 정의

# 전문가 1: 모임 매칭 에이전트 (SubGraph)
def call_meeting_matching_agent(state: MasterAgentState):
    print("--- CALLING: Meeting Matching Agent ---")
    
    # 내부에 자체적인 State, 노드, 그래프를 가진 작은 에이전트
    class MeetingAgentState(TypedDict):
        title: str; description: str; time: str; location: str; query: str;
        context: List[Document]; answer: str; rewrite_count: int; decision: str

    meeting_llm = ChatOpenAI(model="gpt-4o-mini")
    meeting_index_name = os.getenv("PINECONE_INDEX_NAME_MEETING")
    if not meeting_index_name: raise ValueError("'.env' 파일에 PINECONE_INDEX_NAME_MEETING 변수를 설정해야 합니다.")
    embedding_function = OpenAIEmbeddings(model='text-embedding-3-large')
    vector_store = PineconeVectorStore.from_existing_index(index_name=meeting_index_name, embedding=embedding_function)
    retriever = vector_store.as_retriever(search_kwargs={'k': 2})

    prepare_query_prompt = ChatPromptTemplate.from_template(
        "당신은 사용자가 입력한 정보를 바탕으로 유사한 다른 정보를 검색하기 위한 최적의 검색어를 만드는 전문가입니다.\n"
        "아래 [모임 정보]를 종합하여, 벡터 데이터베이스에서 유사한 모임을 찾기 위한 가장 핵심적인 검색 질문을 한 문장으로 만들어주세요.\n"
        "[모임 정보]:\n- 제목: {title}\n- 설명: {description}\n- 시간: {time}\n- 장소: {location}"
    )
    prepare_query_chain = prepare_query_prompt | meeting_llm | StrOutputParser()
    def prepare_query(m_state: MeetingAgentState):
        query = prepare_query_chain.invoke({
            "title": m_state['title'], "description": m_state['description'],
            "time": m_state.get('time', ''), "location": m_state.get('location', '')
        })
        return {"query": query}
    def retrieve(m_state: MeetingAgentState): return {"context": retriever.invoke(m_state['query'])}
    generate_prompt = ChatPromptTemplate.from_template(
        "당신은 MOIT 플랫폼의 친절한 모임 추천 AI입니다. 사용자에게 \"혹시 이런 모임은 어떠세요?\" 라고 제안하는 말투로, "
        "반드시 아래 [검색된 정보]를 기반으로 유사한 모임이 있다는 것을 명확하게 설명해주세요.\n[검색된 정보]:\n{context}\n[사용자 질문]:\n{query}"
    )
    generate_chain = generate_prompt | meeting_llm | StrOutputParser()
    def generate(m_state: MeetingAgentState):
        context = "\n\n".join(doc.page_content for doc in m_state['context'])
        answer = generate_chain.invoke({"context": context, "query": m_state['query']})
        return {"answer": answer}
    check_helpfulness_prompt = ChatPromptTemplate.from_template(
        "당신은 AI 답변을 평가하는 엄격한 평가관입니다. 주어진 [AI 답변]이 사용자의 [원본 질문] 의도에 대해 유용한 제안을 하는지 평가해주세요. "
        "'helpful' 또는 'unhelpful' 둘 중 하나로만 답변해야 합니다.\n[원본 질문]: {query}\n[AI 답변]: {answer}"
    )
    check_helpfulness_chain = check_helpfulness_prompt | meeting_llm | StrOutputParser()
    def check_helpfulness(m_state: MeetingAgentState):
        result = check_helpfulness_chain.invoke({"query": m_state['query'], "answer": m_state['answer']})
        return {"decision": "helpful" if 'helpful' in result.lower() else "unhelpful"}
    rewrite_query_prompt = ChatPromptTemplate.from_template(
        "당신은 사용자의 질문을 더 좋은 검색 결과가 나올 수 있도록 명확하게 다듬는 프롬프트 엔지니어입니다. 주어진 [원본 질문]을 바탕으로, "
        "벡터 데이터베이스에서 더 관련성 높은 모임 정보를 찾을 수 있는 새로운 검색 질문을 하나만 만들어주세요.\n[원본 질문]: {query}"
    )
    rewrite_query_chain = rewrite_query_prompt | meeting_llm | StrOutputParser()
    def rewrite_query(m_state: MeetingAgentState):
        new_query = rewrite_query_chain.invoke({"query": m_state['query']})
        count = m_state.get('rewrite_count', 0) + 1
        return {"query": new_query, "rewrite_count": count}
    
    graph_builder = StateGraph(MeetingAgentState)
    graph_builder.add_node("prepare_query", prepare_query); graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate); graph_builder.add_node("check_helpfulness", check_helpfulness)
    graph_builder.add_node("rewrite_query", rewrite_query)
    graph_builder.set_entry_point("prepare_query"); graph_builder.add_edge("prepare_query", "retrieve")
    graph_builder.add_edge("retrieve", "generate"); graph_builder.add_edge("generate", "check_helpfulness")
    graph_builder.add_conditional_edges("check_helpfulness", lambda state: state['decision'], {"helpful": END, "unhelpful": "rewrite_query"})
    graph_builder.add_edge("rewrite_query", "retrieve")
    meeting_agent = graph_builder.compile()

    user_input = state['user_input']
    initial_state = {
        "title": user_input.get("title", ""), "description": user_input.get("description", ""),
        "time": user_input.get("time", ""), "location": user_input.get("location", ""),
        "rewrite_count": 0
    }
    final_result_state = meeting_agent.invoke(initial_state)
    final_answer = final_result_state.get("answer", "유사한 모임을 찾지 못했습니다.")
    return {"final_answer": final_answer}

# 전문가 2: 취미 추천 에이전트 (Tool)
def call_hobby_recommendation_agent(state: MasterAgentState):
    print("--- CALLING: Hobby Recommendation Agent ---")
    url = "http://127.0.0.1:5000/recommend"
    try:
        response = requests.post(url, json=state['user_input'])
        response.raise_for_status()
        recommendations = response.json()
        if not recommendations:
            final_answer = "아쉽지만 현재 조건에 맞는 취미를 찾지 못했어요."
        else:
            top3 = recommendations[:3]
            answer_parts = ["회원님께는 이런 취미들을 추천해 드려요!\n"]
            for reco in top3:
                answer_parts.append(f"\n- **{reco['name_ko']}**: {reco['short_desc']} (추천 이유: {reco['reason']})")
            final_answer = "".join(answer_parts)
    except requests.exceptions.RequestException as e:
        final_answer = "취미 추천 서버에 문제가 발생하여 연결할 수 없습니다."
    return {"final_answer": final_answer}

# 4-4. 마스터 에이전트 그래프 조립 및 컴파일
master_graph_builder = StateGraph(MasterAgentState)
master_graph_builder.add_node("router", route_request)
master_graph_builder.add_node("meeting_matcher", call_meeting_matching_agent)
master_graph_builder.add_node("hobby_recommender", call_hobby_recommendation_agent)
master_graph_builder.set_entry_point("router")
master_graph_builder.add_conditional_edges(
    "router", lambda state: state['route'],
    {"meeting_matching": "meeting_matcher", "hobby_recommendation": "hobby_recommender"}
)
master_graph_builder.add_edge("meeting_matcher", END)
master_graph_builder.add_edge("hobby_recommender", END)
master_agent = master_graph_builder.compile()

# --- 5. API 엔드포인트 정의 ---
class UserRequest(BaseModel):
    user_input: dict

@app.post("/agent/invoke")
async def invoke_agent(request: UserRequest):
    """사용자의 요청을 받아 마스터 에이전트를 실행하고 결과를 반환합니다."""
    try:
        input_data = {"user_input": request.user_input}
        result = master_agent.invoke(input_data)
        return {"final_answer": result.get("final_answer", "오류: 최종 답변을 생성하지 못했습니다.")}
    except Exception as e:
        print(f"Agent 실행 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="AI 에이전트 처리 중 내부 서버 오류가 발생했습니다.")
    
    
    

# --- Pinecone 업데이트를 위한 API ---

# 1. Pinecone에 저장할 데이터의 형식을 정의합니다.
class NewMeeting(BaseModel):
    meeting_id: str  # 각 모임을 구분할 고유 ID
    title: str
    description: str
    time: str
    location: str

# 2. Pinecone에 데이터를 추가하는 새로운 API 엔드포인트를 정의합니다.
@app.post("/meetings/add")
async def add_meeting_to_pinecone(meeting: NewMeeting):
    """
    새로운 모임 정보를 받아 Pinecone 벡터 DB에 추가(upsert)합니다.
    """
    try:
        print(f"--- Pinecone에 새로운 모임 추가 시작 (ID: {meeting.meeting_id}) ---")
        
        # 1. Pinecone DB에 연결합니다.
        meeting_index_name = os.getenv("PINECONE_INDEX_NAME_MEETING")
        if not meeting_index_name:
            raise ValueError("'.env' 파일에 PINECONE_INDEX_NAME_MEETING이(가) 설정되지 않았습니다.")
            
        embedding_function = OpenAIEmbeddings(model='text-embedding-3-large')
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=meeting_index_name,
            embedding=embedding_function
        )

        # 2. 텍스트 정보를 하나의 문서로 합칩니다.
        # 벡터로 변환할 때는 모든 텍스트 정보를 합쳐서 의미를 풍부하게 만듭니다.
        full_text = (
            f"제목: {meeting.title}\n"
            f"설명: {meeting.description}\n"
            f"시간: {meeting.time}\n"
            f"장소: {meeting.location}"
        )
        
        # 3. 메타데이터를 준비합니다.
        # 메타데이터는 검색 결과로 보여줄 때 사용될 원본 정보입니다.
        metadata = {
            "title": meeting.title,
            "description": meeting.description,
            "time": meeting.time,
            "location": meeting.location,
        }
        
        # 4. Pinecone에 데이터를 추가(upsert)합니다.
        # Pinecone은 'ID'를 기준으로 데이터를 추가하거나 덮어씁니다.
        vector_store.add_texts(
            texts=[full_text],
            metadatas=[metadata],
            ids=[meeting.meeting_id]
        )
        
        print(f"--- Pinecone에 모임 추가 성공 (ID: {meeting.meeting_id}) ---")
        return {"status": "success", "message": f"모임(ID: {meeting.meeting_id})이 성공적으로 추가되었습니다."}

    except Exception as e:
        print(f"Pinecone 업데이트 중 오류 발생: {e}")
        # 오류 발생 시, 백엔드에 어떤 오류인지 알려주어 문제를 해결할 수 있도록 합니다.
        raise HTTPException(status_code=500, detail=f"Pinecone에 모임을 추가하는 중 오류가 발생했습니다: {str(e)}")
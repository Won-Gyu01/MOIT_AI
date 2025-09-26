# MOIT 취미 매칭 소셜 플랫폼

**MOIT(모잇)**는 'AI 전문가 팀'이라는 독자적인 시스템을 통해, 사용자의 성향과 상황에 최적화된 **취미**와 **모임**을 추천하는 플랫폼의 AI 서버입니다. 이 프로젝트는 사용자의 요청을 지능적으로 분석하고 각 분야의 전문 에이전트에게 작업을 분배하는 **멀티 에이전트(Multi-Agent)** 시스템으로 구축되었습니다.

##  주요 특징

* **AI 전문가 팀 아키텍처**: 단일 AI가 아닌, 지능형 라우터(Master Agent)와 각 분야 전문가(Expert Agents)로 구성된 팀이 협업하여 문제를 해결합니다.
* **Self-RAG 기반 모임 추천**: 새로운 모임 생성 시, AI가 먼저 기존 모임 정보를 스스로 검토하고(Retrieve), 추천 결과가 유용한지 판단(Reflect)하여 최적의 결과를 생성(Generate)합니다. 이를 통해 불필요한 모임 생성을 줄이고 커뮤니티 참여를 유도합니다.
* **확장 가능한 에이전트 구조**: `LangGraph`를 사용하여 각 AI 에이전트의 작업 흐름을 모듈화하여, 추후, 새로운 AI 기능을 쉽고 빠르게 추가할 수 있습니다.
* **독립적인 AI 서버**: AI 추천 시스템을 웹 서비스와 완전히 분리하여, AI 연산이 웹 서버에 직접적인 부하를 주지 않도록 설계했습니다.

---

## 시스템 아키텍처 : MOIT의 AI 전문가 팀

MOIT의 AI는 '지능형 지휘관'과 '각 분야 전문가'로 구성된 팀처럼 작동합니다.


1.  **[사용자 요청]**: 사용자가 새로운 모임 생성을 요청합니다.
2.  **[Master Agent]**: '지능형 지휘관' 역할을 하는 Master Agent가 요청의 의도("유사한 모임 찾아줘")를 파악합니다.
3.  **[라우팅]**: Master Agent는 이 작업을 처리할 가장 적합한 전문가인 **'Meeting Match Agent'** 에게 작업을 전달합니다.
4.  **[Meeting Match Agent]**:
    * **Self-RAG 수행**: Pinecone DB에서 유사한 모임을 검색하고, LLM을 통해 추천 결과가 유용한지 스스로 검증합니다.
    * **결과 생성**: 검증된 결과를 바탕으로 사용자에게 제안할 최종 추천 목록을 생성합니다.
5.  **[최종 응답]**: Master Agent는 전문가가 전달한 결과를 받아 사용자에게 최종적으로 응답합니다.

---

##  기술 스택

* **Framework**: `FastAPI`
* **AI / LLM Orchestration**: `LangGraph`, `LangChain`
* **LLM & Embeddings**: `OpenAI (GPT-4o-mini)`, `OpenAI Embeddings`
* **Vector Database**: `Pinecone`
* **Web Server**: `Uvicorn`

---

##  로컬 환경에서 실행 및 테스트하기

### 1단계: 프로젝트 복제 및 환경 설정

```bash
# 1. 저장소 복제
git clone [https://github.com/your-username/moit-project.git](https://github.com/your-username/moit-project.git)
cd moit-project

# 2. 파이썬 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 필요 라이브러리 설치
pip install -r requirements.txt
```

### 2단계: 환경 변수 설정 (`.env` 파일)

프로젝트 루트 디렉토리에 `.env` 파일을 만들고, 아래와 같이 본인의 API 키를 입력하세요.

```env
# .env

# OpenAI API 키
OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"

# Pinecone 벡터 DB API 키 및 인덱스 정보
PINECONE_API_KEY="YOUR_PINECONE_API_KEY_HERE"
PINECONE_INDEX_NAME_MEETING="YOUR_MEETING_INDEX_NAME"
```

### 3단계: FastAPI 서버 실행

터미널에 아래 명령어를 입력하여 로컬 서버를 실행합니다.

```bash
uvicorn main:app --reload
```

서버가 성공적으로 실행되면 터미널에 `Application startup complete.` 메시지와 함께 `http://127.0.0.1:8000` 주소가 나타납니다.

### 4단계: API 기능 테스트하기

웹 브라우저에서 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** 로 접속하세요.

FastAPI가 자동으로 생성해주는 대화형 API 문서 페이지가 나타납니다.

1.  `POST /agent/invoke` 엔드포인트를 열고 `Try it out` 버튼을 클릭하세요.
2.  Request body에 아래와 같이 모임 정보를 입력하고 `Execute` 버튼을 누르면, Master Agent를 통해 AI 전문가 팀이 작동하는 것을 확인할 수 있습니다.

    ```json
    {
      "input": "새로운 모임을 만들고 싶어",
      "meeting_details": {
        "title": "주말에 같이 코딩 공부할 사람?",
        "description": "강남역 근처 카페에서 파이썬 기초 스터디 하실 분 구합니다. 초보자도 환영해요!",
        "category": "IT/개발",
        "tags": ["#코딩", "#파이썬", "#스터디"]
      }
    }
    ```

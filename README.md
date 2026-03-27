> **Note on Repository Structure:**
> This repository houses the **Standalone Core AI Agent Engine (v3.0)** featuring the advanced LangGraph and Gemini multimodal logic. 
> For the full-stack web application integration (Frontend/Backend) managed by our team, please visit the [MOIT Full-Stack Team Repository](https://github.com/taehyeooo/moit-project.git).

# MOIT: Multi-Agent Hobby Matching Platform 

> **AI Agent-Based Hobby Matching Platform for Health Promotion**
> * Gold Prize, The 19th Capstone Design Competition (Soonchunhyang Univ.)*

##  Project Overview
**MOIT** is a personalized hobby matching platform powered by a Multi-Agent system. It is designed to address the limitations of conventional LLM agents—such as **premature termination** and **hallucination**—when dealing with ambiguous user queries. 

By utilizing **LangGraph (StateGraph)**, we engineered a robust architecture that strictly controls the agent's reasoning trajectory and integrates users' implicit contexts (e.g., psychological state, constraints) into the recommendation process.

---

##  My Role & Contributions
**Role:** Lead Developer & AI Agent Architect (`@Won-Gyu01`)

As the team leader and AI architect, I was responsible for designing and implementing the core Agentic AI pipeline:
* **Trajectory Control System:** Designed a deterministic routing architecture using LangGraph to prevent ReAct agents from stopping early without sufficient exploration.
* **Reasoning-Level Personalization (Gemini):** Engineered a multimodal prompt framework that analyzes both user survey data and images to generate highly personalized recommendations.
* **Self-RAG & API Integration:** Built an iterative Retrieval-Augmented Generation (Self-RAG) pipeline combining OpenAI APIs and **Pinecone vector database** to ensure factual consistency.
* **General Search Integration:** Implemented a general search routing fallback using Tavily to handle weather and real-time queries.

---

##  System Architecture & Reasoning Flow

Here are the detailed diagrams illustrating the core system architecture and the reasoning logic of our Multi-Agent system.

### 1. Master Agent Structure (LangGraph Router)
<img width="600" alt="Hobby Recommender Logic (Gemini Multimodal)" src="https://github.com/user-attachments/assets/38a3cd24-5d55-4e97-880d-a8abe559a807" />
> **Figure 1.** High-level overview of the Master Agent routing logic, dispatching user queries to the specialized 'hobby_recommender', 'general_searcher', or 'meeting_matcher' agent.

<br>

### 2. Hobby Recommender Logic (Multimodal Gemini Flow)
<img width="200" alt="Meeting Matcher Reasoning Flow (Self-RAG Loop)" src="https://github.com/user-attachments/assets/2a2f021c-32a3-47b4-9cae-da784a145622" />
> **Figure 2.** Detailed reasoning flow of the Multimodal Hobby Recommender agent, utilizing Google Gemini to analyze both user survey text and profile images for personalized recommendation.

<br>

### 3. Meeting Matcher Reasoning Flow (Self-RAG Loop)
<img width="200" alt="Master Agent Structure Diagram" src="https://github.com/user-attachments/assets/d700d59f-8ef6-4d98-865b-1d3b5752ebeb" />
> **Figure 3.** Detailed view of the Meeting Matcher agent's iterative Retrieval-Augmented Generation (Self-RAG) loop (Retrieve -> Generate -> Reflect -> Rewrite) to find factual existing meeting information.
---

##  Tech Stack
* **AI/Agent Framework:** LangChain, LangGraph, OpenAI API, Google Gemini API, Tavily API
* **Database:** Pinecone (Vector DB)
* **Backend:** FastAPI, Uvicorn
* **Language:** Python

---

##  Getting Started (Local Setup)

### 1. Clone & Environment Setup
```bash
# Clone the repository
git clone [https://github.com/Won-Gyu01/MOIT_AI.git](https://github.com/Won-Gyu01/MOIT_AI.git)
cd MOIT_AI

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

### 2. Environment Variables (`.env`)
Create a `.env` file in the root directory and add the following API keys:
```env
OPENAI_API_KEY="your_openai_api_key"
GOOGLE_API_KEY="your_gemini_api_key"
TAVILY_API_KEY="your_tavily_api_key"
PINECONE_API_KEY="your_pinecone_api_key"
PINECONE_INDEX_NAME_MEETING="your_pinecone_index_name"
```

### 3. Run FastAPI Server
```bash
uvicorn main:app --reload
```
The server will start at `http://127.0.0.1:8000`.

### 4. Test via Swagger UI
1. Navigate to `http://127.0.0.1:8000/docs` in your browser.
2. Open the `POST /agent/invoke` endpoint.
3. Click **Try it out** and test the Master Agent routing logic!

---

##  Publications & Achievements
* **Paper:** "A Study on AI Agent-Based Hobby Matching Platform for Health Promotion" (Expected Dec 2025)
* **Awards:** *  **Gold Prize**, The 19th Capstone Design Competition (Soonchunhyang Univ.)
  *  **Bronze Prize**, 2025 Next-Generation Display Consortium Creative Capstone Design Competition (Chungnam National University, Korea)

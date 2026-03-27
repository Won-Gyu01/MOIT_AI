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
<img src="https://github.com/user-attachments/assets/d8891212-bd48-41f0-91c4-6a324adeb839" width="600" alt="Master Agent Structure Diagram" />
> **Figure 1.** High-level overview of the Master Agent routing logic, dispatching user queries to the specialized 'hobby_recommender', 'general_searcher', or 'meeting_matcher' agent.

<br>

### 2. Hobby Recommender Logic (Multimodal Gemini Flow)
<img src="https://github.com/user-attachments/assets/a6298d18-4111-4f9c-bcb4-6b52fa2f74ee" width="200" alt="Hobby Recommender Logic (Gemini Multimodal)" />
> **Figure 2.** Detailed reasoning flow of the Multimodal Hobby Recommender agent, utilizing Google Gemini to analyze both user survey text and profile images for personalized recommendation.

<br>

### 3. Meeting Matcher Reasoning Flow (Self-RAG Loop)
<img src="https://github.com/user-attachments/assets/fc39bbd7-df22-43c2-be0c-0ea70e68911a" width="200" alt="Meeting Matcher Reasoning Flow (Self-RAG Loop)" />
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

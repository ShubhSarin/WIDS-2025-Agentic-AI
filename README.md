## **Agentic AI for Beginners — WIDS-2025 Submission**

### **Project Overview**
- **Purpose:** A comprehensive learning journey exploring agentic AI concepts, from foundational LLM tasks to production-ready applications.
- **Scope:** Assignments, weekly exercises with Google ADK, and a final YouTube Study Assistant project demonstrating end-to-end agentic AI implementation.

### **Contents**

#### **Assignment 1: Core LLM Capabilities**
- [assignment-1/q1.py](assignment-1/q1.py) - Text summarization with BART
- [assignment-1/q2.py](assignment-1/q2.py) - Text generation with GPT-2
- [assignment-1/q3.py](assignment-1/q3.py) - Sentiment analysis pipeline

#### **Assignment 2: Agentic Orchestration with LangGraph**
- [assignment-2/q1.py](assignment-2/q1.py) - Single-node chat graph with state management
- [assignment-2/q2.py](assignment-2/q2.py) - Two-stage QA pipeline (analyzer → generator)
- [assignment-2/q3.py](assignment-2/q3.py) - Router agent with conditional routing

#### **Week 5: Google ADK (Agent Development Kit)**
- [W5/DataCamp_GoogleADK/](W5/DataCamp_GoogleADK/) - Complete ADK tutorial implementations:
  - **welcome_agent/** - Basic greeting agent with session memory
  - **sequential_agent/** - Chain-based workflow agents
  - **parallel_agent/** - Concurrent execution agents
  - **session_agent/** - Session management and state persistence
  - **structured_agent/** - Structured input/output schemas
  - **tool_agent/** - Custom tools and function calling
  - **persistent_agent/** - Database-backed persistent agents
  - **openai_agent/** - Using OpenAI models via LiteLLM

#### **Final Project: YouTube Study Assistant**
📺 [final/youtube_final_project/](final/youtube_final_project/)

A production-ready Streamlit application that converts YouTube videos/playlists into structured learning materials:
- **Features:**
  - 📘 Chapter-wise study notes generation
  - 🧠 Flashcard creation for memorization
  - 📝 Multiple-choice quiz generation
  - 💬 RAG-based Q&A across all videos
  - 🎬 Playlist support (process entire playlists at once)

- **Tech Stack:**
  - Streamlit for interactive UI
  - Google Gemini (gemini-2.5-flash) for content generation
  - YouTube Transcript API for transcript extraction
  - yt-dlp for playlist parsing

- [requirements.txt](requirements.txt)

### **Tools & Libraries (What was used and why)**
- **Python:** Primary language for experiments and scripts.
- **transformers:** Loading and interacting with pretrained transformer models for language tasks.
- **langchain, langchain-core, langchain-huggingface, langchain-openai, langchain-ollama, langchain-chroma, langchain-community:** Orchestration and agent patterns around LLMs — chains, agents, tool integration, retrieval-augmented generation, and connectors.
- **torch, torchvision, accelerate:** Model runtimes and acceleration for PyTorch-backed models when running locally or fine-tuning.
- **pandas:** Data loading, preprocessing, and simple tabular manipulations used in experiments.
- **langgraph:** Graph-based orchestration and visualization of chains/agent flows (used where workflows are constructed programmatically).
- **langchain-huggingface / langchain-openai / langchain-ollama:** Interfaces to different LLM providers/backends for testing and comparing agent behaviors.
- **chroma (via langchain-chroma):** Local vector store for retrieval-augmented generation experiments.

### **What I learned (Key takeaways)**

#### **Foundational Skills (Assignments 1-2)**
- **Agent basics:** How to structure simple agent loops: prompt → plan → act → observe → refine.
- **Prompting & chaining:** Building reliable prompts, chaining model calls, and splitting complex tasks into sub-steps.
- **LangGraph orchestration:** Graph-based agent workflows with state management, conditional routing, and modular node design.
- **Tool integration:** Connecting LLMs to tools (search, calculators, vector DBs) to extend capabilities beyond pure text completion.
- **Retrieval augmentation:** Using vector stores (Chroma) to provide context and improve factual accuracy.
- **Model backends & tradeoffs:** When to use hosted APIs (OpenAI/Ollama) vs local models (Hugging Face + PyTorch) and performance/cost tradeoffs.

#### **Google ADK Mastery (Week 5)**
- **Code-first agent development:** Building agents with fine-grained control vs. framework abstractions.
- **Agent types:** LLM-based agents, workflow agents (sequential, parallel), and custom agents.
- **Session management:** In-memory vs. persistent state, session services, and runner patterns.
- **Structured outputs:** Enforcing response schemas with Pydantic models for reliable data extraction.
- **Tool calling:** Creating custom tools, using built-in tools, and integrating third-party services.
- **Google Cloud integration:** Deployment-ready patterns with Vertex AI and Google Cloud services.

#### **Production Application (Final Project)**
- **End-to-end architecture:** Designing a full-stack agentic application from UI to LLM backend.
- **Multi-source RAG:** Assembling context from multiple documents (video transcripts) for grounded Q&A.
- **Prompt engineering:** Crafting prompts for specific output formats (notes, flashcards, quizzes).
- **State management:** Session persistence in Streamlit across user interactions.
- **API integration:** Working with YouTube APIs, transcript extraction, and playlist handling.

#### **Framework Comparison**
- **LangChain/LangGraph:** Flexible, multi-provider, graph-based orchestration with extensive tooling.
- **Google ADK:** Google Cloud-native, code-first, with built-in deployment and tighter Gemini integration.
- **Trade-offs:** Understanding when to use each framework based on deployment target, cloud ecosystem, and complexity needs.

### **How to run**

#### **Assignments**
- **Install dependencies:**

```bash
python -m pip install -r requirements.txt
```

- **Run a single script:**

```bash
python assignment-1/q1.py
python assignment-2/q1.py
```

#### **Google ADK Agents (Week 5)**

1. **Set up Google Cloud API Key:**
   - Create a project in [Google Cloud Console](https://console.cloud.google.com/)
   - Get an API key from [Google AI Studio](https://aistudio.google.com/) (free tier available)
   - Create a `.env` file in the agent directory with `GOOGLE_API_KEY=your_key_here`

2. **Run agents:**
```bash
# Using ADK CLI (interactive terminal)
cd W5/DataCamp_GoogleADK/welcome_agent
adk run

# Using ADK Web UI
adk web
```

#### **Final Project: YouTube Study Assistant**

1. **Navigate to project directory:**
```bash
cd final/youtube_final_project
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up API key:**
   - Create `.env` file with `GOOGLE_API_KEY=your_gemini_api_key`

4. **Run the Streamlit app:**
```bash
streamlit run streamlit_app.py
```

5. **Usage:**
   - Paste a YouTube video URL or playlist URL
   - Click "Process" to extract transcripts
   - Generate notes, flashcards, or quizzes for each video
   - Ask questions about the content using the chat interface

- **Notes:**
  - Some packages (e.g., `torch`) in `requirements.txt` reference custom index URLs; follow printed installation output if additional steps are required.
  - If using an external LLM provider (OpenAI/Ollama), set provider credentials as environment variables per the provider docs.
  - For Google ADK agents, ensure you have created a `.env` file with your `GOOGLE_API_KEY`.

### **Project Structure**
```
WIDS-2025/
├── assignment-1/          # Core LLM tasks (summarization, generation, sentiment)
├── assignment-2/          # LangGraph agent orchestration
├── W5/
│   └── DataCamp_GoogleADK/    # Google ADK agent implementations
│       ├── welcome_agent/
│       ├── sequential_agent/
│       ├── parallel_agent/
│       ├── session_agent/
│       ├── structured_agent/
│       ├── tool_agent/
│       ├── persistent_agent/
│       └── openai_agent/
├── final/
│   └── youtube_final_project/  # Production YouTube Study Assistant
│       ├── streamlit_app.py
│       ├── core/
│       │   ├── llm.py
│       │   ├── transcript.py
│       │   ├── summarizer.py
│       │   ├── flashcards.py
│       │   ├── quiz.py
│       │   └── rag.py
│       └── requirements.txt
├── requirements.txt       # Root dependencies
└── README.md
```

### **Documentation**
📄 For a comprehensive overview of the entire learning journey, see the [LaTeX Report](report/report.tex) which includes:
- Detailed analysis of each assignment
- In-depth LangChain and LangGraph framework coverage
- Google ADK concepts and implementations
- Final project architecture and design decisions
- Framework comparison and trade-offs

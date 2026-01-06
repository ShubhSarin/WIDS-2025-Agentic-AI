## **Agentic AI for Beginners — WIDS-2025 Submission**

### **Project Overview**
- **Purpose:** A beginner-friendly collection of exercises exploring agentic AI concepts, tool use, and simple agent workflows.
- **Scope:** Two assignment folders each containing three scripts (`q1.py`, `q2.py`, `q3.py`) that demonstrate tasks and experiments with language models and agentic patterns.

### **Contents**
- **Files:**
  - [assignment-1/q1.py](assignment-1/q1.py)
  - [assignment-1/q2.py](assignment-1/q2.py)
  - [assignment-1/q3.py](assignment-1/q3.py)
  - [assignment-2/q1.py](assignment-2/q1.py)
  - [assignment-2/q2.py](assignment-2/q2.py)
  - [assignment-2/q3.py](assignment-2/q3.py)
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
- **Agent basics:** How to structure simple agent loops: prompt → plan → act → observe → refine.
- **Prompting & chaining:** Building reliable prompts, chaining model calls, and splitting complex tasks into sub-steps.
- **Tool integration:** Connecting LLMs to tools (search, calculators, vector DBs) to extend capabilities beyond pure text completion.
- **Retrieval augmentation:** Using vector stores (Chroma) to provide context and improve factual accuracy.
- **Model backends & tradeoffs:** When to use hosted APIs (OpenAI/Ollama) vs local models (Hugging Face + PyTorch) and performance/cost tradeoffs.
- **Safety and guardrails:** The importance of prompt-level checks, constraints, and verification steps when building agentic systems.
- **Orchestration libraries:** How LangChain and related libraries simplify pipeline construction and let you experiment with agent types quickly.

### **How to run**
- **Install dependencies:**

```bash
python -m pip install -r requirements.txt
```

- **Run a single script:**

```bash
python assignment-1/q1.py
```

- **Notes:**
  - Some packages (e.g., `torch`) in `requirements.txt` reference custom index URLs; follow printed installation output if additional steps are required.
  - If using an external LLM provider (OpenAI/Ollama), set provider credentials as environment variables per the provider docs.



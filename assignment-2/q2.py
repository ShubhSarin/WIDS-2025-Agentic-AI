from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer  
from langchain_community.llms import HuggingFacePipeline
import torch

class QAState(TypedDict):
    messages: List[Dict[str,str]]
    clarified_question: str
    final_answer: str
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

pipe_question_analyzer = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    max_new_tokens=64,
    temperature = 0.2,
    do_sample = True,
)

llm_question_analyzer = HuggingFacePipeline(pipeline=pipe_question_analyzer)

pipe_answer_generator = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    max_new_tokens=256,
    temperature = 0.7,
    do_sample = True,
)

llm_answer_generator = HuggingFacePipeline(pipeline=pipe_answer_generator)
def build_prompt(messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

SYSTEM_PROMPT_QUESTION_ANALYZER = """Rewrite the user's question clearly and simply.\n\n
    Rules:\n
    - Output ONLY the rewritten question.\n
    - Output exactly ONE sentence.\n
    - Do NOT explain.\n
    - Do NOT answer.\n
    - Do NOT repeat instructions.\n
    - If 'LLM' appears, assume it means 'Large Language Model'."""

def clean_output(text: str) -> str:
    text = text.strip()

    for token in ["<|system|>", "<|user|>", "<|assistant|>"]:
        text = text.replace(token, "").strip()

    # collapse whitespace
    text = " ".join(text.split())

    return text

def question_analyzer(state: QAState):
    message = [
        {
            "role": "system", 
         "content": SYSTEM_PROMPT_QUESTION_ANALYZER
        },
        {
            "role": "user", 
            "content": state["messages"][-1]["content"]
        }
    ]
    prompt = build_prompt(message)
    response = llm_question_analyzer.invoke(prompt)
    return {
        "clarified_question": response
    }
def answer_generator(state: QAState):
    message = [
        {"role": "system", "content": "You answer questions clearly and helpfully."},
        {"role": "user", "content": state["clarified_question"]}
    ]
    prompt = build_prompt(message)
    response = llm_answer_generator.invoke(prompt)
    return {
        "final_answer": response,
        "messages": state["messages"] + [
            {"role": "assistant", "content": response}
        ]
    }

graph = StateGraph(QAState)

graph.add_node("analyzer", question_analyzer)
graph.add_node("answerer", answer_generator)

graph.set_entry_point("analyzer")
graph.add_edge("analyzer", "answerer")
graph.set_finish_point("answerer")

app = graph.compile()

state = {
    "messages": [
        {"role": "user", "content": "pls explain large language models easily"}
    ],
    "clarified_question": "",
    "final_answer": ""
}

state = app.invoke(state)

print("Clarified question:")
print(state["clarified_question"])

print("\nFinal answer:")
print(state["final_answer"])
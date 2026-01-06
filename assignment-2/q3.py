from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_community.llms import HuggingFacePipeline
import torch

class RouterState(TypedDict):
    messages: List[Dict[str,str]]
    route: str
    final_answer: str

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

def build_prompt(messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

pipe_router = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=10,
    do_sample=False,
    temperature=0.0,
)

llm_router = HuggingFacePipeline(pipeline=pipe_router)

pipe_python = pipeline(
    "text-generation",
    model=model,
    max_new_tokens=256,
    temperature=0.4,
    do_sample=True,
    tokenizer=tokenizer,
)
llm_python = HuggingFacePipeline(pipeline=pipe_python)

pipe_general = pipeline(
    "text-generation",
    model=model,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
    tokenizer=tokenizer,
)
llm_general = HuggingFacePipeline(pipeline=pipe_general)

def clean_output(text: str) -> str:
    for token in ["<|im_start|>", "<|im_end|>", "system", "user", "assistant"]:
        text = text.replace(token, "")
    return " ".join(text.split())

ROUTER_SYSTEM_PROMPT = """
    You are a router.\n
    Decide which agent should answer the user's question.\n\n
    Respond with EXACTLY ONE WORD on the LAST LINE.\n
    Valid outputs:\n
    PYTHON\n
    GENERAL\n\n
    Rules:\n
    - PYTHON: programming, coding, Python language, libraries.\n
    - GENERAL: everything else, including animals.
"""
    
def router_node(state: RouterState):
    user_msg = state["messages"][-1]["content"]
    
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg}
    ]
    
    prompt = build_prompt(messages)
    raw = llm_router.invoke(prompt)
    
    cleaned = raw.splitlines()[-1].strip()
    
    if cleaned not in {"PYTHON", "GENERAL"}:
        cleaned = "GENERAL"
    
    
    decision = cleaned
    
    return {
        "route": decision
    }

def python_agent(state: RouterState):
    messages = [
        {"role": "system", "content": "You are a python expert"},
        {"role": "user", "content": state["messages"][-1]["content"]},
    ]
    
    prompt = build_prompt(messages)
    answer = llm_python.invoke(prompt)
    answer = clean_output(answer)
    
    return {
        "final_answer": answer,
        "messages": state["messages"] + [
            {"role": "assistant", "content": answer}
        ]
    }

def general_agent(state: RouterState):
    messages = [
        {"role": "system", "content": "You answer general questions clearly."},
        {"role": "user", "content": state["messages"][-1]["content"]},
    ]
    
    prompt = build_prompt(messages)
    answer = llm_general.invoke(prompt)
    answer = clean_output(answer)
    
    return {
        "final_answer": answer,
        "messages": state["messages"] + [
            {"role": "assistant", "content": answer}
        ]
    }
    
graph = StateGraph(RouterState)

graph.add_node("router", router_node)
graph.add_node("python", python_agent)
graph.add_node("general", general_agent)

graph.set_entry_point("router")
graph.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "PYTHON": "python",
        "GENERAL": "general",
    }
)
graph.set_finish_point("python")
graph.set_finish_point("general")

app = graph.compile()

#test
state = {
    "messages": [
        {"role": "user", "content": "How do I reverse a list in python"}
    ],
    "route": "",
    "final_answer": "",
}

state = app.invoke(state)

print("Route: ", state["route"])
print("Python Answer: ", state["final_answer"])


#test 2
state = {
    "messages": [
        {"role": "user", "content": "what ingrediants do i need to cook alfredo pasta"}
    ],
    "route": "",
    "final_answer": "",
}

state = app.invoke(state)

print("Route: ", state["route"])
print("General Answer: ", state["final_answer"])

from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer  
from langchain_community.llms import HuggingFacePipeline
import torch


pipe = pipeline(
    "text-generation", 
    model = "Qwen/Qwen2.5-1.5B-Instruct",
    device=0 if torch.cuda.is_available() else -1,
    max_new_tokens = 512,
)

llm = HuggingFacePipeline(pipeline=pipe)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

class ChatState(TypedDict):
    messages: List[Dict[str, str]]
    
def build_prompt(messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True
    )
    
def llm_node(state: ChatState):
    prompt = build_prompt(state["messages"])
    output = llm.invoke(prompt)
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": output}]
    }
    
graph = StateGraph(ChatState)
graph.add_node("llm", llm_node)
graph.set_entry_point("llm")
graph.set_finish_point("llm")
app = graph.compile()

state = {
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        }
    ]
}
#query 1
state["messages"].append({"role": "user", "content": "Explain what's inheritence in object oriented programming"})
state = app.invoke(state)
print(state["messages"][-1]["content"])

#query2
state["messages"].append({"role": "user", "content": "Explain what's riemann integration"})
state = app.invoke(state)
print(state["messages"][-1]["content"])

#query3
state["messages"].append({"role": "user", "content": "List some places to visit in mumbai"})
state = app.invoke(state)
print(state["messages"][-1]["content"])
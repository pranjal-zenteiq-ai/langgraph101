from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
import os
load_dotenv()
from langgraph.graph import StateGraph,START,END
from typing import Dict

class State(Dict):
    message:str
    caption:str

def llm1(state:State)->State:
    client=ChatNVIDIA(
        model=os.getenv("NVIDIA_MODEL"),
        api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0.3,
        top_p=1,
        max_completion_tokens=16384,
        model_kwargs={"thinking":False},
    )
    prompt=f"Write a post for my instagram on the topic {state['message']}"
    output=client.invoke(prompt)
    return {"message":output.content}

def llm2(state:State)->State:
    client=ChatNVIDIA(
        model=os.getenv("NVIDIA_MODEL"),
        api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0.3,
        top_p=1,
        max_completion_tokens=16384,
        model_kwargs={"thinking":False},
    )
    prompt=f"""{state['message']}\n\n this is the post i wrote for instagrma, can you give me a witty caption for this?"""
    output=client.invoke(prompt)
    return {"caption":output.content}
g=StateGraph(State)
g.add_node("llm1",llm1)
g.add_node("llm2",llm2)
g.add_edge(START,"llm1")
g.add_edge("llm1","llm2")
g.add_edge("llm2",END)

workflow=g.compile()
print(workflow.invoke({"message":"What is so special about maharashtra? "}))
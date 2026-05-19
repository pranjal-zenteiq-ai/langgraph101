from langgraph.graph import StateGraph,START,END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.types import Command
from dotenv import load_dotenv
import os

load_dotenv()

class State(TypedDict):
    message:str
    response:str
    review:str

client=ChatNVIDIA(
    model=os.getenv("NVIDIA_MODEL"),
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.5,
    top_p=1,
    max_completion_tokens=16384,
)

def llm(state:State)->State:
    prompt=f"You are a helpful assistant. This is the user query: {state['message']}"
    response=""
    for chunk in client.stream(prompt):
        print(chunk.content,end="",flush=True)
        response+=chunk.content
    print()
    return {"response":response}

def hitl(state):
    decision=interrupt({
        "response":state["response"],
        "question":"approve, reject, or edit?"
    })
    return {"review":decision}

def route_after_hitl(state:State):
    if state["review"]=="reject":
        return "llm"
    return END

g=StateGraph(State)
g.add_node("llm",llm)
g.add_node("hitl",hitl)
g.add_edge(START,"llm")
g.add_edge("llm","hitl")
g.add_conditional_edges("hitl",route_after_hitl)
workflow=g.compile(checkpointer=MemorySaver())
config={"configurable":{"thread_id":"1"}}

workflow.invoke({"message":"Is maharashtra the graetest state of them all?"},config)

while True:
    state=workflow.get_state(config)
    if not state.next:
        break
    human=input("Your review: ")
    workflow.invoke(Command(resume=human),config)
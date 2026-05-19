from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
from dotenv import load_dotenv
load_dotenv()
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

client = ChatNVIDIA(
    model=os.getenv("NVIDIA_MODEL"),
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.3,
    top_p=1,
    max_completion_tokens=16384,
)

def chat(state: State) -> dict:
    output = client.invoke(state["messages"])
    return {"messages": [output]}

g = StateGraph(State)
g.add_node("chat", chat)
g.add_edge(START, "chat")
g.add_edge("chat", END)

workflow = g.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "1"}}
print("--- Turn 1 ---")
res1 = workflow.invoke(
    {"messages": [("user", "Hello! My name is Pranjal.")]},
    config
)
print("Assistant:", res1["messages"][-1].content)

print("\n--- Turn 2 ---")
res2 = workflow.invoke(
    {"messages": [("user", "What is my name?")]},
    config
)
print("Assistant:", res2["messages"][-1].content)
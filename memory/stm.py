from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from typing import TypedDict,Annotated
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
import os
load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm=ChatNVIDIA(
    model=os.getenv("NVIDIA_MODEL"),
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.7,
    top_p=1,
    max_completion_tokens=16384
)

def agent(state:State):
    response=llm.invoke(state["messages"])
    return {"messages":[response]}

g=StateGraph(MessagesState)
g.add_node("agent",agent)
g.add_edge(START,"agent")
g.add_edge("agent",END)

from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = g.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "thread-1"}}
config2 = {"configurable": {"thread_id": "thread-2"}}
graph.invoke({"messages": [{"role": "user", "content": "Hi! My name is Pranjal."}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config)
print(graph.get_state(config).values)
# print(graph.get_state(config2).values)

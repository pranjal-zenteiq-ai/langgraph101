from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import StateGraph,START,END
from typing import TypedDict
import os
from dotenv import load_dotenv
load_dotenv()

class State(TypedDict):
    message:str

def chat_llm(state:State)->dict:
    client=ChatNVIDIA(
        model=os.getenv("NVIDIA_MODEL"),
        api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0.3,
        top_p=1,
        max_completion_tokens=16384,
    )
    input=state["message"]
    prompt=f"""you are a helpful assistant. This is the user query {input}, you have to answer that correctly and precisely. 
    For fun whatever you answer just add something in marathi at the end.
    """
    output=client.invoke(prompt)
    return {"message":output.content}

g=StateGraph(State)
g.add_node("chat_llm",chat_llm)
g.add_edge(START,"chat_llm")
g.add_edge("chat_llm",END)

workflow=g.compile()
print(workflow.invoke({"message":"What is so special about maharashtra? "}))


from langgraph.prebuilt import tool_node,tools_condition
from langgraph.graph import StateGraph,START,END
from langchain_core.tools import tool
from dotenv import load_dotenv
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from typing import TypedDict
load_dotenv()
import os
from langgraph.prebuilt import ToolNode,tools_condition
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage

class State1(TypedDict):
    english:str
    marathi:str

class State2(TypedDict):
    input:str
    english:str
    marathi:str

llm=ChatNVIDIA(
    model=os.getenv("NVIDIA_MODEL"),
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.7,
    top_p=1,
    max_completion_tokens=16384
)

def translation(state:State1)->State1:
    prompt=f"Translate the following text from english to marathi\n\n{state['english']}"
    client=llm.invoke(prompt)
    return {"marathi":client.content}

def generate(state:State2)->State2:
    prompt=f"""
    Generate a poem on the given topic in English
    {state['input']}
    """
    client=llm.invoke(prompt)
    return {"english":client.content}

g2=StateGraph(State1)
g2.add_node("translation",translation)
g2.add_edge(START,"translation")
g2.add_edge("translation",END)
workflow2=g2.compile()

g=StateGraph(State2)
g.add_node("generate",generate)
g.add_node("translation_subgraph",workflow2)
g.add_edge(START,"generate")
g.add_edge("generate","translation_subgraph")
g.add_edge("translation_subgraph",END)

workflow=g.compile()

inputs={"input":"a rainy day in the city"}
result=workflow.invoke(inputs)

print(result["english"])
print(result["marathi"])
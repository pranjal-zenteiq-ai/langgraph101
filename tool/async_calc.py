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


class State(TypedDict):
    messages: Annotated[list, add_messages]

llm=ChatNVIDIA(
    model=os.getenv("NVIDIA_MODEL"),
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.7,
    top_p=1,
    max_completion_tokens=16384,
    model_kwargs={"thinking":False}
)

@tool
def calculator(a:int,b:int)->float:
    '''
    This function is used to perform addition of two numbers
    '''
    return a+b

@tool
def weather(city:str)->str:
    '''
    This function is used to get the weather of a city
    '''
    import random 
    from random import randint
    return f"The weather in {city} is {randint(0,100)}% sunny"

tools=[calculator,weather]

llm_tools=llm.bind_tools(tools)

def agent(state: State):
    response = llm_tools.invoke(state["messages"])
    return {"messages": [response]}
    

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


class State(TypedDict):
    messages: Annotated[list, add_messages]

llm=ChatNVIDIA(
    model=os.getenv("NVIDIA_MODEL"),
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.7,
    top_p=1,
    max_completion_tokens=16384,
)

@tool
def calculator(a:int,b:int)->float:
    '''
    This function is used to perform addition of two numbers
    '''
    return a+b

@tool
def weather(city:str)->str:
    '''
    This function is used to get the weather of a city
    '''
    import random 
    from random import randint
    return f"The weather in {city} is {randint(0,100)}% sunny"

tools=[calculator,weather]

llm_tools=llm.bind_tools(tools)

async def agent(state: State):
    response = await llm_tools.ainvoke(state["messages"])
    return {"messages": [response]}
    
g=StateGraph(State)
g.add_node("agent",agent)
g.add_node("tool_node",ToolNode(tools))

g.add_edge(START,"agent")
g.add_conditional_edges("agent",tools_condition,{"tools":"tool_node","__end__":END})
g.add_edge("tool_node","agent")
workflow=g.compile()
import asyncio

async def main():
    result=await workflow.ainvoke({
        "messages":[
            HumanMessage(content="What is the weather in pune?")
        ]
    })
    print(result["messages"][-1].content)

asyncio.run(main())

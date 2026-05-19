from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Literal
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
from dotenv import load_dotenv
load_dotenv()
import random
class State(TypedDict):
    paper:str
    mark1:int
    mark2:int
    result:Literal["pass","fail"]
    motivation:str|None

def checker1(state:State)->State:
    state['mark1']=random.randint(0,10)
    return {"mark1":state['mark1']}

def checker2(state:State)->State:
    state['mark2']=(random.randint(0,10)*random.randint(0,10))%11
    return {"mark2":state['mark2']}

def result_checker(state:State)->State:
    if state['mark1']+state['mark2']>=15:
        state['result']="pass"
    else:
        state['result']="fail"
    return state

def motivator(state:State)->State:
    llm=ChatNVIDIA(
        model=os.getenv("NVIDIA_MODEL"),
        api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0.5,
        top_p=1,
        max_completion_tokens=16384,
    )
    prompt=f"You are a teacher and your student just failed their exam. Write a small motivating paragraph in marathi."
    output=llm.invoke(prompt)
    state['motivation']=output.content
    return state

g=StateGraph(State)
g.add_node("checker1",checker1)
g.add_node("checker2",checker2)
g.add_node("result checker",result_checker)
g.add_node("motivator",motivator)
g.add_edge(START,"checker1")
g.add_edge(START,"checker2")
g.add_edge("checker1","result checker")
g.add_edge("checker2","result checker")
g.add_conditional_edges(
    "result checker",
    lambda x: "motivator" if x['result']=="fail" else END
)
g.add_edge("motivator",END)
workflow=g.compile()
print(workflow.invoke({"paper":"Hi my name is Pranjal"}))

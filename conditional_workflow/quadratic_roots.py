from langgraph.graph import StateGraph,START,END
import math 
from typing import TypedDict

class State(TypedDict):
    a:int
    b:int
    c:int
    d:float
    root1:float
    root2:float
    
def det(state:State)->State:
    state['d']=state['b']**2-4*state['a']*state['c']
    return state
    
def pos_root(state:State)->State:
    state['root1']=(-state['b']+math.sqrt(state['d']))/(2*state['a'])
    state['root2']=(-state['b']-math.sqrt(state['d']))/(2*state['a'])
    return state

def not_pos_root(state:State)->State:
    state['root1']='Not applicable'
    state['root2']='Not applicable'
    return state

def zero(state:State)->State:
    state['root1']=-state['b']/(2*state['a'])
    state['root2']=-state['b']/(2*state['a'])
    return state

g=StateGraph(State)
g.add_node("det",det)
g.add_node("pos_root",pos_root)
g.add_node("not_pos_root",not_pos_root)
g.add_node("zero",zero)
g.add_edge(START,"det")
g.add_conditional_edges("det",
    lambda x:"pos_root" if x['d']>0 else "not_pos_root" if x['d']<0 else "zero",
    ["pos_root","not_pos_root","zero"]
)
g.add_edge("pos_root",END)
g.add_edge("not_pos_root",END)
g.add_edge("zero",END)
workflow=g.compile()
print(workflow.invoke({"a":int(input("enter a: ")),"b":int(input("enter b: ")),"c":int(input("enter c: "))}))
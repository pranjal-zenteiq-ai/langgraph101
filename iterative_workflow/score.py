from langgraph.graph import StateGraph,START,END
from typing import TypedDict
import random
class State(TypedDict):
    score:int
    result:str
    count:int
    
def score(state:State)->State:
    state['score']=random.randint(0,100)
    return state

def prime_checker(state:State)->State:
    x=state['score']
    for i in range (2,x):
        if x%i==0 and x!=i:
            state['result']='Not prime'
            return state
    state['result']='prime'
    state['count']+=1
    return state

g=StateGraph(State)
g.add_node("score",score)
g.add_node("prime_checker",prime_checker)
g.add_edge(START,"score")
g.add_edge("score","prime_checker")
g.add_conditional_edges(
    "prime_checker",
    lambda x: END if x['result']=='prime' else "score"
)
g.add_edge("prime_checker",END)
workflow=g.compile()
print(workflow.invoke({"count":0}))
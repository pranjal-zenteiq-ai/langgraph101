from langgraph.graph import StateGraph,START,END
from typing import TypedDict
import random
class State(TypedDict):
    TeamA_Player:str
    TeamB_Player:str
    TeamA_Score:int
    TeamB_Score:int
    result:str
    
def score_teamA(state:State)->State:
    state['TeamA_Score']=random.randint(0,5)
    return state

def score_teamB(state:State)->State:
    state['TeamB_Score']=random.randint(0,5)
    return state

def player_teamA(state:State)->State:
    state['TeamA_Player']=input("Enter the name of player from Team A: ")
    return state

def player_teamB(state:State)->State:
    state['TeamB_Player']=input("Enter the name of player from Team B: ")
    return state

def result(state:State)->State:
    if state['TeamA_Score']>state['TeamB_Score']:
        state['result']=f'{state['TeamA_Player']} won the match'
    elif state['TeamA_Score']<state['TeamB_Score']:
        state['result']=f'{state['TeamB_Player']} won the match'
    else:
        state['result']='Draw'
    return state

g=StateGraph(State)
g.add_node("score_teamA",score_teamA)
g.add_node("score_teamB",score_teamB)
g.add_node("player_teamA",player_teamA)
g.add_node("player_teamB",player_teamB)
g.add_node("result",result)
g.add_edge(START,"score_teamA")
g.add_edge(START,"score_teamB")
g.add_edge(START,"player_teamA")
g.add_edge(START,"player_teamB")
g.add_edge("score_teamA","result")
g.add_edge("score_teamB","result")
g.add_edge("player_teamA","result")
g.add_edge("player_teamB","result")
g.add_edge("result",END)
workflow=g.compile()
print(workflow.invoke({}))
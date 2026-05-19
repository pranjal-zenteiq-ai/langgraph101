from langgraph.graph import StateGraph,START,END
from typing import Dict
from IPython.display import Image
class BaseState(Dict):
    height:float
    weight:float
    bmi:float
    category:str



def bmi(state:BaseState)->BaseState:
    bmi=state["weight"]/(state["height"]**2)
    state["bmi"]=bmi
    return state

def category(state:BaseState)->BaseState:
    if state["bmi"]<18.5:
        state["category"]="Underweight"
    elif state["bmi"]<25:
        state["category"]="Normal weight"
    elif state["bmi"]<30:
        state["category"]="Overweight"
    else:
        state["category"]="Obesity"
    return state
graph=StateGraph(BaseState)
graph.add_node("bmi",bmi)
graph.add_node("category",category)
graph.add_edge(START,"bmi")
graph.add_edge("bmi","category")
graph.add_edge("category",END)

workflow=graph.compile()

print(workflow.invoke({"height":1.75,"weight":70}))
#workflow.get_graph().draw_ascii()
Image(workflow.get_graph().draw_mermaid_png())
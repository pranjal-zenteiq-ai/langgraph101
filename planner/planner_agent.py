import os
import json
from typing import TypedDict
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import StateGraph,START,END
load_dotenv()

class State(TypedDict):
 task:str
 plan:list[str]
 index:int
 trace:str
 final:str

client=ChatNVIDIA(
 model=os.getenv("NVIDIA_MODEL"),
 api_key=os.getenv("NVIDIA_API_KEY"),
 temperature=0.3,
 top_p=1,
 max_completion_tokens=16384,
)

def planner(state:State)->State:
 prompt=f"""you are a planner agent.
return only valid json exactly like this:
{{"plan":["step1","step2","step3"]}}
task:{state["task"]}
"""
 output=client.invoke(prompt)
 print(output.content)
 data=json.loads(output.content)
 return {
  "plan":data["plan"],
  "index":0,
  "trace":"",
  "final":""
 }

def step(state:State)->State:
 current=state["plan"][state["index"]]
 print(current)
 prompt=f"""execute this step only.
task:{state["task"]}
step:{current}
trace:{state["trace"]}
return short output only.
"""
 output=client.invoke(prompt)
 print(output.content)
 trace=state["trace"]+f"\n{current}\n{output.content}\n"
 return {
  "trace":trace,
  "index":state["index"]+1
 }

def check(state:State):
 if state["index"]>=len(state["plan"]):
  return "final"
 return "step"

def final(state:State)->State:
 prompt=f"""combine this into final answer.
task:{state["task"]}
trace:{state["trace"]}
write final answer.
"""
 output=client.invoke(prompt)
 print(output.content)
 return {"final":output.content}

g=StateGraph(State)
g.add_node("planner",planner)
g.add_node("step",step)
g.add_node("final",final)
g.add_edge(START,"planner")
g.add_edge("planner","step")
g.add_conditional_edges("step",check,{"step":"step","final":"final"})
g.add_edge("final",END)
workflow=g.compile()

if __name__=="__main__":
#  result=workflow.invoke({
#   "task":"plan a simple way to learn langgraph in 3 days",
#   "plan":[],
#   "index":0,
#   "trace":"",
#   "final":""
#  })
#  print(result["final"])
 for chunks,metadata in workflow.stream(
    {"task":"plan a trip to goa"},
    config={"configurable":{"thread_id":"1"}},
    stream_mode="messages"
    ):
    if chunks.content:
        print(chunks.content,end="",flush=True)
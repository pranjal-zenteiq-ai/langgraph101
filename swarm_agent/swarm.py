from typing import Annotated,TypedDict
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
load_dotenv()
import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__) 

class State(TypedDict):
    messages:Annotated[list,add_messages]
    active_agent:str
    nxt_agent:str

llm=ChatNVIDIA(
    model=os.getenv("NVIDIA_MODEL"),
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.7,
    top_p=1,
    max_completion_tokens=16384
)

def last_user_text(state:State):
    return next((m.content.lower() for m in reversed(state["messages"]) if isinstance(m,HumanMessage)),"")

def route(state:State):
    text=last_user_text(state)
    agent=state.get("active_agent","caption_node")
    if "translate" in text:
        # logger.info(f"route chose translate_node")
        agent="translate_node"
    elif "poem" in text:
        # logger.info(f"route chose poem_node")
        agent="poem_node"
    elif "caption" in text:
        # logger.info(f"route chose caption_node")
        agent="caption_node"
    logger.info(f"route chose {agent}")
    return agent

def caption_node(state:State):
    text=last_user_text(state)
    msg=llm.invoke([SystemMessage(content="You are caption agent. Write captions.")] + state["messages"])
    next_agent="translate_node" if "translate" in text else ""
    logger.info(f"caption_node chose next_agent: {next_agent}")
    return {"messages":[msg],"active_agent":"caption_node","nxt_agent":next_agent}

def poem_node(state:State):
    text=last_user_text(state)
    msg=llm.invoke([SystemMessage(content="You are poem agent. Write poems.")] + state["messages"])
    next_agent="translate_node" if "translate" in text else ""
    logger.info(f"poem_node chose next_agent: {next_agent}")
    return {"messages":[msg],"active_agent":"poem_node","nxt_agent":next_agent}

def translate_node(state:State):
    text=last_user_text(state)
    msg=llm.invoke([SystemMessage(content="You are translation agent. Translate clearly.")] + state["messages"])
    next_agent="poem_node" if "poem" in text else "caption_node" if "caption" in text else ""
    logger.info(f"translate_node chose next_agent: {next_agent}")
    return {"messages":[msg],"active_agent":"translate_node","nxt_agent":next_agent}

def choose_next(state:State):
    text=last_user_text(state)
    logger.info(f"choose_next chose: {state.get('nxt_agent',END) or END}")
    return state.get("nxt_agent",END) or END

graph=StateGraph(State)
graph.add_node("caption_node",caption_node)
graph.add_node("poem_node",poem_node)
graph.add_node("translate_node",translate_node)
graph.add_conditional_edges(
    START,
    route,
    {
        "caption_node":"caption_node",
        "poem_node":"poem_node",
        "translate_node":"translate_node"
    }
)
graph.add_conditional_edges(
    "caption_node",
    choose_next,
    {
        "poem_node":"poem_node",
        "translate_node":"translate_node",
        END:END
    }
)
graph.add_conditional_edges(
    "poem_node",
    choose_next,
    {
        "caption_node":"caption_node",
        "translate_node":"translate_node",
        END:END
    }
)
graph.add_conditional_edges(
    "translate_node",
    choose_next,
    {
        "caption_node":"caption_node",
        "poem_node":"poem_node",
        END:END
    }
)
app=graph.compile(checkpointer=MemorySaver())

if __name__=="__main__":
    config={"configurable":{"thread_id":"demo-1"}}
    print("starting")

    out1=app.invoke(
        {"messages":[HumanMessage(content="Write a caption for a rainy evening.")]},
        config
    )
    print(out1["messages"][-1].content)

    out2=app.invoke(
        {"messages":[HumanMessage(content="Now make it a poem.")]},
        config
    )
    print(out2["messages"][-1].content)

    out3=app.invoke(
        {"messages":[HumanMessage(content="Translate that into Marathi.")]},
        config
    )
    print(out3["messages"][-1].content)
    out4=app.invoke(
        {"messages":[HumanMessage(content="Improve this a bit more and keep the same meaning.")]},
        config
    )
    print(out4["messages"][-1].content)
    print(out4["active_agent"]) 
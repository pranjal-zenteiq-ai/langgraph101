from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.graph import StateGraph,START,END
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from sentence_transformers import SentenceTransformer
from typing import TypedDict,Annotated
from dotenv import load_dotenv
import os
import numpy as np
from pypdf import PdfReader
load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm=ChatNVIDIA(
    model=os.getenv("NVIDIA_MODEL"),
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.7,
    top_p=1,
    max_completion_tokens=16384
)

embed_model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embedding(text:str)->list[float]:
    vec=embed_model.encode(text)
    return vec.tolist()

def load_pdf_chunks(pdf_path:str)->list[str]:
    reader=PdfReader(pdf_path)

    text=""
    for page in reader.pages:
        page_text=page.extract_text()
        if page_text:
            text+=page_text+"\n"

    chunks=[]
    chunk_size=1000

    for i in range(0,len(text),chunk_size):
        chunks.append(text[i:i+chunk_size])

    return chunks

pdf_path="/home/zenteiq/Downloads/stem_topiccc.pdf"

chunks=load_pdf_chunks(pdf_path)

chunk_vectors=[np.array(embedding(chunk)) for chunk in chunks]

def cosine_sim(a,b):
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

@tool
def fetch_chunks(query:str)->str:
    """
    Fetch the most relevant chunks for a query.
    """
    qvec=np.array(embedding(query))
    scores=[]
    for i,vec in enumerate(chunk_vectors):
        scores.append((cosine_sim(qvec,vec),chunks[i]))
    scores.sort(reverse=True,key=lambda x:x[0])
    top=[text for _,text in scores[:2]]
    return "\n".join(top)

tools=[fetch_chunks]
llm_tools=llm.bind_tools(tools)

def agent(state:State):
    response=llm_tools.invoke(state["messages"])
    return {"messages":[response]}

g=StateGraph(State)
g.add_node("agent",agent)
g.add_node("tools",ToolNode(tools))

g.add_edge(START,"agent")
g.add_conditional_edges("agent",tools_condition)
g.add_edge("tools","agent")

workflow=g.compile()

result=workflow.invoke({
    "messages":[HumanMessage(content="What is this pdf about?")]
})

print(result["messages"][-1].content)
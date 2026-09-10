from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from typing import Annotated, TypedDict
import operator
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

load_dotenv()

model = init_chat_model(
    "google_genai:gemini-3.1-flash-lite",
    temperature=0.3,
)

class CourtState(TypedDict):
    case: str
    transcript: Annotated[list[str], operator.add]
    verdict: str

def prosecutor(state: CourtState) -> dict:
    reply = model.invoke([
        SystemMessage(content="당신은 법정의 검사이다. 3문장으로 말해라."),
        HumanMessage(content=f"사건:{state['case']}")
    ])
    return {"transcript":[f"검사:{reply.text}\n"]}

def defense(state: CourtState) -> dict:
    record = "\n".join(state['transcript'])
    reply = model.invoke([
        SystemMessage(content="당신은 법정의 변호사이다. 3문장으로 말해라."),
        HumanMessage(content=f"사건:{state['case']}\n지금까지의 공판:{record}")
    ])
    return {"transcript":[f"변호사:{reply.text}\n"]}

def judge(state: CourtState) -> dict:
    record = "\n".join(state['transcript'])
    reply = model.invoke([
        SystemMessage(content="당신은 법정의 판사이다. 유죄 또는 무죄 중 하나를 고르고 이유를 2문장으로 말해라."),
        HumanMessage(content=f"사건:{state['case']}\n공판 기록:{record}"),
    ])
    return {
        "transcript":[f"판사:{reply.text}\n"],
        "verdict":reply.text
    }

builder = StateGraph(CourtState)
builder.add_node("prosecutor",prosecutor)
builder.add_node("defense",defense)
builder.add_node("judge",judge)

builder.add_edge(START,"prosecutor")
builder.add_edge("prosecutor","defense")
builder.add_edge("defense","judge")
builder.add_edge("judge",END)

graph = builder.compile()

case = input("어떤 사건인가요?: ")
if case:
    result = graph.invoke({
        "case": case,
        "transcript": [],
        "verdict": "",
    })
    print("\n".join(result["transcript"]))
    print("---")
    print("판결:", result["verdict"])

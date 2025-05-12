from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.constants import Send

from elasticsearch import Elasticsearch

from typing import Literal, Annotated, List, TypedDict
import operator
import asyncio


# --- LLM Initialization ---
llm = ChatOllama(
    model="llama3.1:8b",
    base_url="http://localhost:11434"
)

# --- Example Documents to Summarize ---
documents = [
    Document(page_content="What is Python?", metadata={"title": "python_book"}),
    # Document(page_content="Blueberries are blue", metadata={"title": "blueberry_book"}),
    # Document(page_content="Bananas are yellow", metadata={"title": "banana_book"}),
]


# --- Prompts ---
map_template = """Answer the following question: {content}"""
reduce_template = """
The following is a set of responses:
{docs}
Take these and distill it into a final, consolidated result
of the main themes.
"""
map_prompt = ChatPromptTemplate.from_messages([("human", map_template)])
reduce_prompt = ChatPromptTemplate.from_messages([("human", reduce_template)])
reduce_chain = reduce_prompt | llm | StrOutputParser()


# --- Graph State Definitions ---
class OverallState(TypedDict):
    contents: List[str]
    summaries: Annotated[list, operator.add]
    final_summary: str

# class SummaryState(TypedDict):
#     content: str
#     messages: Annotated[List[AIMessage], operator.add]

es = Elasticsearch("http://localhost:9200", headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=9"})
INDEX_NAME = "docs_index"

@tool
def retrieve_context(query: str):
    """
    Searches for relevant documents in Elasticsearch and returns a string of relevant content.

    Args:
        query (str): The search query to use in Elasticsearch.

    Returns:
        str: A string containing the relevant content from Elasticsearch.
    """

    response = es.search(index=INDEX_NAME, query={"match": {"content": query}}, size=5)
    results = [hit["_source"]["content"] for hit in response["hits"]["hits"]]
    return "\n".join(results)

# --- Tool Node ---
tools = [retrieve_context]
# tool_node = ToolNode(tools)
model = llm.bind_tools(tools)

tools_by_name = {tool.name: tool for tool in tools}
def tool_node(state: MessagesState):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

# --- Map: Summary Generation Node ---
async def generate_summary(state: MessagesState):
    human_msg = HumanMessage(content=map_template.format(content=state["messages"]))
    response = await model.ainvoke([human_msg])
    return {
        "messages": [response],
        "summaries": [response] 
    }

# --- Condition: Should Continue with Tool? ---
def should_continue(state: MessagesState) -> Literal["tools", "generate_final_summary"]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "generate_final_summary"


# --- Final Reduce Step ---
async def generate_final_summary(state: OverallState):
    summaries = [
        msg.content.strip() for msg in state["summaries"]
        if isinstance(msg, AIMessage)
    ]
    response = await reduce_chain.ainvoke({"docs": "\n".join(summaries)})
    return {"final_summary": response}


# --- Mapping Function ---
def map_summaries(state: OverallState):
    return [
        Send("generate_summary", {"messages": content})
        for content in state["contents"]
    ]

# --- Graph Construction ---
graph = StateGraph(OverallState)
graph.add_node("generate_summary", generate_summary)
graph.add_node("generate_final_summary", generate_final_summary)
graph.add_node("tools", tool_node)

graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
graph.add_conditional_edges("generate_summary", should_continue)
graph.add_edge("tools", "generate_summary")
graph.add_edge("generate_final_summary", END)

app = graph.compile()

# --- Execution ---
async def run_graph():
    async for step in app.astream({"contents": [doc.page_content for doc in documents]}):
        print(step)

asyncio.run(run_graph())

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.constants import Send

from elasticsearch import Elasticsearch

from typing import Literal, Annotated, List, TypedDict, Union
import operator
import asyncio

# --- LLM Initialization ---
llm = ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434")

# --- Elasticsearch Setup ---
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

tools = [retrieve_context]
model = llm.bind_tools(tools)
tools_by_name = {tool.name: tool for tool in tools}

def tool_node(state: MessagesState):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

# --- Prompts ---
map_template = """Answer the following question or analyze the content:\n\n{content}"""
map_prompt = ChatPromptTemplate.from_messages([("human", map_template)])
map_chain = map_prompt | model

reduce_template = """The following is a set of analyses:\n{docs}\n\nCombine these into a final, consolidated summary of the main themes."""
reduce_prompt = ChatPromptTemplate.from_messages([("human", reduce_template)])
reduce_chain = reduce_prompt | llm | StrOutputParser()

# --- State Definitions ---
class OverallState(TypedDict):
    initial_input: str
    contents: List[HumanMessage]
    summaries: Annotated[list, operator.add]
    final_summary: str

# --- Summary Generation ---
async def generate_summary(state: MessagesState):
    print(f"\n\nGENERATE\n\n{state}")
    response = await model.ainvoke(state["messages"])
    if isinstance(response, AIMessage) and not response.tool_calls:
        return {
            "messages": [response],
            "summaries": [response.content]
        }
    else:
        return {
            "messages": [response]
        }


def should_continue(state: MessagesState) -> Literal["tools", "generate_final_summary"]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "generate_final_summary"

# --- Final Reduce Step ---
async def generate_final_summary(state: OverallState):
    print(f"\n\n===REDUCE===\n\n{state}\n\n")
    summaries = "\n".join(state["summaries"])
    print(f"\n\n{summaries}\n\n")
    response = await reduce_chain.ainvoke({"docs": summaries})
    return {"final_summary": response}

# --- Split large input and return list of strings ---
def preprocess_input(state: OverallState):
    paragraphs = [p.strip() for p in state["initial_input"].split("\n\n") if p.strip()]
    return {"contents": paragraphs}

def map_summaries(state: OverallState):
    return [Send("generate_summary", {"messages": [msg]}) for msg in state["contents"]]

# --- Graph Construction ---
graph = StateGraph(OverallState)
graph.add_node("generate_summary", generate_summary)
graph.add_node("generate_final_summary", generate_final_summary)
graph.add_node("tools", tool_node)
graph.add_node("preprocess_input", preprocess_input)

graph.add_edge(START, "preprocess_input")
graph.add_conditional_edges("preprocess_input", map_summaries, ["generate_summary"])
graph.add_conditional_edges("generate_summary", should_continue)
graph.add_edge("tools", "generate_summary")
graph.add_edge("generate_final_summary", END)

app = graph.compile()

# --- Execution ---
async def run_graph():
    large_input_text = """
Where is Lebron James from?

How old is Lebron James?
"""
    async for step in app.astream({"initial_input": large_input_text}):
        print(f"\n\n{step}\n\n")

asyncio.run(run_graph())

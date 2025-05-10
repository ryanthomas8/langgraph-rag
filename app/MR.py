from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from langgraph.graph import END, START, StateGraph
from langgraph.constants import Send
from langgraph.prebuilt import ToolNode

from elasticsearch import Elasticsearch
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from typing import Literal, Annotated, List, TypedDict
import operator
import asyncio
from uuid import uuid4


# --- LLM Initialization ---
llm = ChatOllama(
    model="gemma2:latest",
    base_url="http://localhost:11434"
)


# --- Example Documents to Summarize ---
documents = [
    Document(page_content="Apples are red", metadata={"title": "apple_book"}),
    Document(page_content="Blueberries are blue", metadata={"title": "blueberry_book"}),
    Document(page_content="Bananas are yellow", metadata={"title": "banana_book"}),
]


# --- Prompts ---
map_template = """Write a concise summary of the following: {context}.
    Leverage the retrieve_context tool to get more information.
    """
reduce_template = """
The following is a set of summaries:
{docs}
Take these and distill it into a final, consolidated summary
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

class SummaryState(TypedDict):
    content: str
    messages: Annotated[List[AIMessage], operator.add]


# --- Elasticsearch Tool ---
es = Elasticsearch("http://localhost:9200")
INDEX_NAME = "docs_index"

@tool
def retrieve_context(query: str):
    """Search for relevant documents in Elasticsearch."""
    urls = [
        "https://docs.python.org/3/tutorial/index.html",
        "https://realpython.com/python-basics/",
        "https://www.learnpython.org/"
    ]
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    doc_splits = text_splitter.split_documents(docs)

    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(index=INDEX_NAME)

    for doc in doc_splits:
        es.index(index=INDEX_NAME, id=str(uuid4()), document={"content": doc.page_content})

    response = es.search(index=INDEX_NAME, query={"match": {"content": query}}, size=5)
    results = [hit["_source"]["content"] for hit in response["hits"]["hits"]]
    return "\n".join(results)


# --- Tool Node ---
tools = [retrieve_context]
tool_node = ToolNode(tools)
model = llm.bind_tools(tools)

# --- Map: Summary Generation Node ---
async def generate_summary(state: SummaryState):
    human_msg = HumanMessage(content=map_template.format(context=state["content"]))
    response = await model.ainvoke([human_msg])
    return {
        "messages": [response],
        "summaries": [response] 
    }


# --- Condition: Should Continue with Tool? ---
def should_continue(state: SummaryState) -> Literal["tools", "generate_final_summary"]:
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
        Send("generate_summary", {"content": content, "messages": []})
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

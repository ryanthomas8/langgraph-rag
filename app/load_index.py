from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from uuid import uuid4
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200", headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=9"})
INDEX_NAME = "docs_index"

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
    
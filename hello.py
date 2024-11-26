from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.llms.ollama import Ollama
import chromadb
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

local_llm = Ollama(model="llama3.2")

# Embedding Model to do local embedding using Ollama.
ollama_embedding = OllamaEmbedding("llama3.2")
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

# url = "https://raw.githubusercontent.com/elastic/elasticsearch-labs/main/datasets/workplace-documents.json"
documents = SimpleDirectoryReader("./resource").load_data()
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=ollama_embedding
)

# Customer Query
query = "What are the organizations sales goals?"
bundle = QueryBundle(
    query_str=query, embedding=ollama_embedding.get_query_embedding(query=query))

query_engine = index.as_query_engine(llm=local_llm)
response = query_engine.query(bundle)

print("response:", response.response)

import logging
from typing import Optional

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import (
    VectorStoreIndex,
    QueryBundle,
    StorageContext,
    SimpleDirectoryReader
)
from llama_index.llms.ollama import Ollama
import chromadb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentQuery:
    """A class to handle document querying using LlamaIndex."""
    
    def __init__(
        self,
        model_name: str = "llama3.2",
        data_dir: str = "./resource",
        collection_name: str = "quickstart"
    ):
        """
        Initialize the DocumentQuery with the specified configuration.
        
        Args:
            model_name: Name of the Ollama model to use
            data_dir: Directory containing the documents to index
            collection_name: Name of the ChromaDB collection
        """
        self.model_name = model_name
        self.data_dir = data_dir
        self.collection_name = collection_name
        
        # Initialize components
        self.local_llm = Ollama(model=model_name)
        self.ollama_embedding = OllamaEmbedding(model_name)
        self.index = self._create_index()
        
    def _create_index(self) -> VectorStoreIndex:
        """Create and return the vector store index."""
        try:
            logger.info("Loading documents from %s", self.data_dir)
            documents = SimpleDirectoryReader(self.data_dir).load_data()
            
            logger.info("Creating ChromaDB collection: %s", self.collection_name)
            chroma_client = chromadb.EphemeralClient()
            chroma_collection = chroma_client.create_collection(self.collection_name)
            
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            return VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.ollama_embedding
            )
        except Exception as e:
            logger.error("Error creating index: %s", str(e))
            raise
    
    def query(self, query_text: str) -> Optional[str]:
        """
        Query the document index.
        
        Args:
            query_text: The query string to search for
            
        Returns:
            The response text or None if an error occurs
        """
        try:
            logger.info("Processing query: %s", query_text)
            bundle = QueryBundle(
                query_str=query_text,
                embedding=self.ollama_embedding.get_query_embedding(query=query_text)
            )
            
            query_engine = self.index.as_query_engine(llm=self.local_llm)
            response = query_engine.query(bundle)
            
            return response.response
        except Exception as e:
            logger.error("Error processing query: %s", str(e))
            return None

def main():
    """Main function to demonstrate document querying."""
    try:
        # Initialize the document query system
        doc_query = DocumentQuery()
        
        # Example query
        query = "What are the organizations sales goals?"
        logger.info("Executing query: %s", query)
        
        response = doc_query.query(query)
        if response:
            print("\nResponse:", response)
        else:
            logger.error("No response received for the query")
            
    except Exception as e:
        logger.error("Application error: %s", str(e))

if __name__ == "__main__":
    main()

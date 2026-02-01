import logging
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Initialize logger for this module
logger = logging.getLogger(__name__)


class VectorManager:
    def __init__(self, chunks):
        """
        Initializes the vector store with processed text chunks and embedding model.
        """
        logger.info("Initializing VectorManager with text chunks.")
        self.chunks = chunks
        self.embeddings = OpenAIEmbeddings()  # Uses text-embedding-3-small by default
        self.vector_store = None

    def create_and_save_index(self, index_path="faiss_index"):
        """
        Converts chunks to embeddings and saves them locally to a FAISS index.
        """
        if not self.chunks:
            logger.warning("No chunks provided to create_and_save_index. Skipping.")
            return

        try:
            # Extract only the text for embedding, but keep metadata for retrieval
            logger.info(f"Extracting text and metadata for {len(self.chunks)} chunks.")
            texts = [chunk["text"] for chunk in self.chunks]
            metadatas = [{"source": chunk["source"]} for chunk in self.chunks]

            # Convert texts to numerical embeddings and build the FAISS index
            logger.info("Generating embeddings and building FAISS vector store. This may take a moment...")
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )

            # Save index locally to avoid re-embedding every time
            self.vector_store.save_local(index_path)
            logger.info(f"Vector store successfully saved to local directory: {index_path}")

        except Exception as e:
            logger.error(f"Failed to create or save vector index: {e}", exc_info=True)

    def get_retriever(self):
        """
        Returns a retriever object to be used by the Agent for semantic search.
        """
        if self.vector_store:
            logger.info("Creating retriever with search parameter k=5.")
            return self.vector_store.as_retriever(search_kwargs={"k": 5})

        logger.error("Attempted to get retriever, but vector_store is not initialized.")
        return None
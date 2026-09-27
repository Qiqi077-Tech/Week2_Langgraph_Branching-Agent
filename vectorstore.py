from dotenv import load_dotenv
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore

load_dotenv()

INDEX_NAME = "apple-10k"
EMBEDDING_MODEL = "multilingual-e5-large"
EMBEDDING_DIMENSION = 1024


def get_vector_store() -> PineconeVectorStore:
    embeddings = PineconeEmbeddings(model=EMBEDDING_MODEL)
    return PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

import chromadb
from chromadb.utils import embedding_functions


def get_collection(collection_name: str = "hillsborough_tracts"):
    client = chromadb.PersistentClient(path="./vectorstore")
    ef = embedding_functions.DefaultEmbeddingFunction()
    return client.get_collection(name=collection_name, embedding_function=ef)


def retrieve_context(query: str, n_results: int = 8) -> str:
    """Semantic search over ChromaDB for tracts relevant to the query."""
    try:
        collection = get_collection()
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )

        docs = results.get("documents", [[]])[0]
        if not docs:
            return "No relevant context found in the vector database."

        context_lines = [f"{i + 1}. {doc}" for i, doc in enumerate(docs)]
        return "Top relevant census tracts from semantic search:\n" + "\n".join(context_lines)

    except Exception as e:
        return f"Vector search unavailable: {e}. Proceeding with spatial-only results."
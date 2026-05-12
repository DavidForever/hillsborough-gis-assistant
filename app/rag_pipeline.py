import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

VECTORSTORE_PATH = "vectorstore/faiss_index.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

_index = None
_documents = None
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def load_vectorstore():
    global _index, _documents
    if _index is None and os.path.exists(VECTORSTORE_PATH):
        with open(VECTORSTORE_PATH, "rb") as f:
            data = pickle.load(f)
            _index = data["index"]
            _documents = data["documents"]
    return _index, _documents


def retrieve_context(query: str, n_results: int = 8) -> str:
    try:
        index, documents = load_vectorstore()
        if index is None or documents is None:
            return "Vector store not found. Proceeding with spatial results only."

        model = get_model()
        query_embedding = model.encode([query]).astype("float32")
        faiss.normalize_L2(query_embedding)

        distances, indices = index.search(query_embedding, n_results)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(documents):
                results.append(f"{i+1}. {documents[idx]}")

        if not results:
            return "No relevant context found."

        return "Top relevant census tracts:\n" + "\n".join(results)

    except Exception as e:
        return f"Vector search unavailable: {e}. Proceeding with spatial results only."
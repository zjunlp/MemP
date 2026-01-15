import ast
import time
import pickle
import numpy as np

def compute_facts_embeddings(documents, embedding):
    """
    Compute embeddings for facts in each document.

    :param documents: list of Document objects.
    :param embedding: embedding model with embed_query method.
    :return: dict mapping source -> {fact_key: embedding}
    """
    result = {}
    for doc in documents:
        while True:
            try:
                # doc_facts = ast.literal_eval(doc.metadata.get('facts', '{}'))
                facts_value = doc.metadata.get('facts', '{}')
                if isinstance(facts_value, str):
                    try:
                        doc_facts = ast.literal_eval(facts_value)
                    except Exception as e:
                        print(f"[ERROR] Failed to parse facts: {facts_value}, error: {e}")
                        doc_facts = {}
                elif isinstance(facts_value, dict):
                    doc_facts = facts_value
                else:
                    print(f"[WARN] Unexpected type for facts: {type(facts_value)}")
                    doc_facts = {}

                fact_embeddings = {
                    k: embedding.embed_query(str(v))
                    for k, v in doc_facts.items()
                }
                result[doc.metadata['source']] = fact_embeddings
                break
            except Exception as e:
                print(f"Error parsing doc {doc.metadata.get('source')}: {e}")
                # Retry
                time.sleep(1)
    return result

def save_facts_embedding_cache(path, doc_facts_embeddings):
    with open(path, "wb") as f:
        pickle.dump(doc_facts_embeddings, f)
    print(f"Saved facts embedding cache to {path}")

def load_facts_embedding_cache(path):
    import os
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors.

    :param vec1: The first vector.
    :param vec2: The second vector.
    :return: The cosine similarity score.
    """
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

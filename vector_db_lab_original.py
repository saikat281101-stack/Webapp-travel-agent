"""Travel Agent Vector Database with ChromaDB.

Creates/recreates the ChromaDB collection used by rag_travel_agent.py.
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DATA_FILE = "travel_agent_data.txt"
DB_DIR = "./chroma_db"
COLLECTION = "travel_agent"


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text, chunk_size=500, overlap=80):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def main():
    text = load_text(DATA_FILE)
    chunks = chunk_text(text)

    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=DB_DIR)

    # Recreate the collection so the database always matches the current KB.
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=embedding_fn,
    )

    collection.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[
            {
                "source": DATA_FILE,
                "chunk": i,
                "domain": "travel",
            }
            for i in range(len(chunks))
        ],
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB.")
    print(f"Database: {DB_DIR}")
    print(f"Collection: {COLLECTION}")

    # Optional test query
    query = input("\nAsk a travel question to test retrieval (or press Enter to skip): ").strip()

    if query:
        results = collection.query(
            query_texts=[query],
            n_results=min(3, len(chunks)),
        )

        print("\n--- Top matching chunks ---")

        for i, (doc, distance) in enumerate(
            zip(results["documents"][0], results["distances"][0]),
            start=1,
        ):
            print(f"\nResult {i} | distance={distance:.4f}")
            print(doc)


if __name__ == "__main__":
    main()

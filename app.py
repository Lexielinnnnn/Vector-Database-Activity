"""Science Search: Streamlit + SciQ + Sentence Transformers + Chroma.

Run: python3.11 -m streamlit run app.py
The first launch downloads the public dataset/model and builds the index.
"""
from pathlib import Path
from threading import Lock
import logging

import chromadb
import streamlit as st
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "sciq_app_v1"
RECORD_LIMIT = 1000


def chunk_text(text, tokenizer, size=200, overlap=30):
    token_ids = tokenizer.encode(
        text, add_special_tokens=False, truncation=False, verbose=False
    )
    start = 0
    while start < len(token_ids):
        end = min(start + size, len(token_ids))
        yield tokenizer.decode(token_ids[start:end], skip_special_tokens=True)
        if end == len(token_ids):
            break
        start = end - overlap


@st.cache_resource
def load_resources():
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )
    metadata = collection.metadata or {}
    expected_count = metadata.get("expected_count", -1)
    if (
        metadata.get("ready") is True
        and expected_count > 0
        and collection.count() == expected_count
    ):
        return model, collection, Lock()

    dataset = load_dataset("allenai/sciq", split="train")
    dataset = dataset.filter(
        lambda row: bool((row["support"] or "").strip())
    )
    dataset = dataset.select(range(min(RECORD_LIMIT, len(dataset))))
    chunks, metadatas, chunk_ids = [], [], []
    for record_id, row in enumerate(dataset):
        text = (
            f"Question: {row['question']}\n"
            f"Correct answer: {row['correct_answer']}\n"
            f"Explanation: {row['support']}"
        )
        for chunk_index, chunk in enumerate(chunk_text(text, model.tokenizer)):
            chunks.append(chunk)
            metadatas.append({
                "record_id": record_id,
                "chunk_index": chunk_index,
                "question": row["question"],
                "correct_answer": row["correct_answer"],
            })
            chunk_ids.append(f"record_{record_id}_chunk_{chunk_index}")

    if not chunks:
        raise ValueError("No science passages were found.")

    # Recover safely from an interrupted first build.
    existing_ids = collection.get(include=[])["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)
    collection.modify(metadata={
        "ready": False, "expected_count": len(chunks)
    })
    for start in range(0, len(chunks), 128):
        end = start + 128
        embeddings = model.encode(
            chunks[start:end], batch_size=32,
            normalize_embeddings=True, show_progress_bar=False,
        )
        collection.upsert(
            ids=chunk_ids[start:end], documents=chunks[start:end],
            embeddings=embeddings.tolist(), metadatas=metadatas[start:end],
        )
    if collection.count() != len(chunks):
        raise RuntimeError("The search index did not finish building.")
    collection.modify(metadata={
        "ready": True,
        "expected_count": len(chunks),
        "record_count": len(dataset),
    })
    return model, collection, Lock()


def search(query, top_k, model, collection, lock):
    # Serialize access to the model shared by Streamlit sessions.
    with lock:
        embedding = model.encode([query], normalize_embeddings=True).tolist()
        count = collection.count()
        candidate_count = min(top_k * 10, count)
        while True:
            results = collection.query(
                query_embeddings=embedding,
                n_results=candidate_count,
                include=["documents", "metadatas", "distances"],
            )
            seen, matches = set(), []
            for document, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                record_id = metadata["record_id"]
                if record_id in seen:
                    continue
                seen.add(record_id)
                matches.append({
                    "document": document,
                    "metadata": metadata,
                    "similarity": 1 - distance,
                })
                if len(matches) == top_k:
                    break
            if len(matches) >= top_k or candidate_count >= count:
                return matches
            candidate_count = min(candidate_count * 2, count)


def main():
    st.set_page_config(page_title="Science Search", page_icon="🔬")
    st.title("🔬 Science Search")
    st.write("Find science knowledge by meaning, not just keywords.")
    st.caption("Search 1,000 science records from the SciQ dataset.")

    try:
        with st.spinner(
            "Preparing search engine… The first launch downloads the model "
            "and dataset, then builds the search index."
        ):
            model, collection, lock = load_resources()
    except Exception:
        logging.exception("Search engine initialization failed")
        st.error(
            "The search engine could not start. Check the terminal or cloud "
            "logs for details, then retry."
        )
        if st.button("Retry initialization"):
            load_resources.clear()
            st.rerun()
        st.stop()

    if collection.count() == 0:
        st.error("The search index is empty.")
        st.stop()

    with st.expander("Example questions"):
        st.write("- How do plants get energy from sunlight?")
        st.write("- Why do objects fall toward the ground?")
        st.write("- What organisms are used to make cheese and yogurt?")

    with st.form("search_form"):
        query = st.text_input(
            "Enter a science question",
            placeholder="How do plants get energy from sunlight?",
            max_chars=1000,
        )
        top_k = st.selectbox("Number of results", [3, 5, 10], index=1)
        submitted = st.form_submit_button("Search")

    if submitted:
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Searching…"):
                    matches = search(
                        query.strip(), top_k, model, collection, lock
                    )
                st.session_state["last_search"] = {
                    "query": query.strip(), "matches": matches,
                }
            except Exception:
                logging.exception("Search failed")
                st.session_state.pop("last_search", None)
                st.error("Search failed. Check the logs and try again.")

    previous = st.session_state.get("last_search")
    if previous:
        st.write("**Results for:**", previous["query"])
        for rank, match in enumerate(previous["matches"], start=1):
            metadata = match["metadata"]
            with st.container(border=True):
                st.subheader(f"{rank}. {metadata['question']}")
                st.write("**Dataset answer:**", metadata["correct_answer"])
                st.write("**Matched passage:**")
                st.write(match["document"])
                st.caption(
                    f"Cosine similarity: {match['similarity']:.3f} "
                    f"• SciQ record: {metadata['record_id']}"
                )
        st.caption(
            "These are related SciQ records, not generated answers to your "
            "question. Similarity measures relevance, not factual confidence. "
            "Even the closest matches may be unrelated."
        )

    st.markdown("[Dataset: SciQ](https://huggingface.co/datasets/allenai/sciq)")


if __name__ == "__main__":
    main()


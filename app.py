import math
import streamlit as st

from generate_answer import generate_answer


st.set_page_config(page_title="CFA Portfolio RAG", page_icon="📘", layout="wide")

st.title("CFA Portfolio Management RAG")
st.caption("Grounded Q&A over CFA Kaplan portfolio materials")

with st.sidebar:
    st.header("Retrieval Settings")
    strategy = st.selectbox("Chunking Strategy", ["semantic", "fixed", "recursive"], index=0)
    semantic_weight = st.slider("Semantic Weight", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    bm25_weight = round(1.0 - semantic_weight, 1)
    st.write(f"BM25 Weight: {bm25_weight}")
    top_k = st.slider("Top-K Retrieval", min_value=5, max_value=12, value=8, step=1)

query = st.text_input("Ask a portfolio management question", placeholder="What is the efficient frontier?")
run_search = st.button("Search", type="primary")

if run_search and query.strip():
    with st.spinner("Retrieving and generating answer..."):
        result = generate_answer(
            query=query.strip(),
            top_k=top_k,
            strategy=strategy,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
        )

    st.subheader("Answer")
    st.write(result.get("answer", ""))

    conf_raw = result.get("confidence", {}).get("rerank_mean", 0.0)
    conf_score = 1 / (1 + math.exp(-conf_raw))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence Score", f"{conf_score:.2f}")
    c2.metric("Backend", result.get("generation_backend", "unknown"))
    c3.metric("Contexts Used", str(result.get("confidence", {}).get("num_supporting_contexts", 0)))
    c4.metric("Total Latency (s)", f"{result.get('latency', {}).get('total_sec', 0.0):.2f}")

    st.subheader("Sources")
    sources = result.get("sources", [])
    if not sources:
        st.info("No source metadata available.")
    else:
        for i, src in enumerate(sources, start=1):
            source_name = src.get("source", "Unknown source")
            topic = src.get("topic", "")
            rerank_score = src.get("rerank_score", 0.0)
            retrieval_score = src.get("retrieval_score", 0.0)
            preview = src.get("preview", "")

            with st.expander(f"Source {i}: {source_name} | Topic: {topic}"):
                st.write(preview)
                st.caption(
                    f"Retrieval score: {retrieval_score:.3f} | Rerank score: {rerank_score:.3f}"
                )

    st.subheader("Manual Grounding Check")
    st.write("- Verify answer claims appear in the source previews.")
    st.write("- If not found, the expected response is: 'Information not found in dataset'.")

elif run_search:
    st.warning("Please enter a question first.")

"""
Health RAG Assistant — Streamlit Frontend

UI decisions documented inline:
- Confidence % and Signal label merged into one metric (they were the same
  information shown twice with different formats).
- Progress bars now use relevance_score (cosine similarity, 0–1 range)
  instead of the broken RRF score / 0.05 calculation.
- Expanded query shown as an info banner only when expansion actually fired.
- Retrieval strategy removed from user-facing UI (internal detail).
- "Retrieval Quality" replaces the raw confidence % — more meaningful label.
- Sources panel redesigned: shows readable relevance % not raw float score.
"""

import time

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Health RAG Assistant",
    page_icon="🏥",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏥 Health RAG")

    st.markdown("### 📄 Knowledge Base")
    st.write("India's Health Transformation (PIB)")

    st.markdown("---")

    st.markdown("### 🤖 Model")
    st.write("Llama 3.3 70B (Groq)")

    st.markdown("### 🔎 Retrieval Pipeline")
    st.write("• Dense: FAISS + Sentence Transformers")
    st.write("• Sparse: BM25 Okapi")
    st.write("• Fusion: Reciprocal Rank Fusion (RRF)")
    st.write("• Diversity: MMR (λ=0.7)")
    st.write("• Ranking: Metadata-aware intent boosting")
    st.write("• Expansion: Abbreviation + synonym injection")

    st.markdown("---")

    st.markdown("### ⚙️ Stack")
    st.write("FastAPI · FAISS · Streamlit · Groq")

    st.markdown("---")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------
st.title("🏥 Health Policy RAG Assistant")
st.caption(
    "Ask questions about India's National Health Schemes. "
    "Powered by Hybrid Dense + Sparse Retrieval with Groq Llama 3.3 70B."
)

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
question = st.chat_input("Ask about PM-JAY, ABDM, infrastructure, schemes...")

if not question:
    st.stop()

# Show user message
st.session_state.messages.append({"role": "user", "content": question})
with st.chat_message("user"):
    st.markdown(question)

# ---------------------------------------------------------------------------
# Backend call
# ---------------------------------------------------------------------------
with st.chat_message("assistant"):
    with st.spinner("Searching knowledge base..."):
        try:
            resp = requests.post(
                "http://127.0.0.1:8000/query",
                json={"question": question},
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.ConnectionError:
            st.error(
                "⚠️ Cannot connect to the FastAPI backend. "
                "Make sure `uvicorn app.main:app --reload` is running."
            )
            st.stop()
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
            st.stop()

    # -----------------------------------------------------------------------
    # Typing animation
    # -----------------------------------------------------------------------
    placeholder = st.empty()
    answer = result["answer"]
    displayed = ""
    for word in answer.split():
        displayed += word + " "
        placeholder.markdown(displayed)
        time.sleep(0.018)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    st.divider()

    # -----------------------------------------------------------------------
    # Query expansion banner — only shown when expansion actually fired
    # -----------------------------------------------------------------------
    query_expanded = result.get("query_expanded", "")
    if query_expanded and query_expanded.strip() != question.strip():
        st.info(
            f"🔍 **Abbreviation detected** — searched for: *{query_expanded}*"
        )

    # -----------------------------------------------------------------------
    # Metrics row
    # -----------------------------------------------------------------------
    label = result.get("confidence_label", "Low")
    confidence_pct = result.get("confidence", 0.0)

    # Map label to emoji for a single clean metric
    label_display = {"High": "🟢 High", "Medium": "🟡 Medium", "Low": "🔴 Low"}.get(
        label, label
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        # "Retrieval Quality" is more meaningful to non-engineers than "Confidence %"
        st.metric("Retrieval Quality", label_display)

    with col2:
        # Source count — useful because it tells the user how many passages
        # were found, which varies by query type.
        st.metric("Sources Found", len(result.get("sources", [])))

    with col3:
        st.metric("Response Time", f"{result.get('response_time_ms', 0):.0f} ms")

    # -----------------------------------------------------------------------
    # Sources panel
    # -----------------------------------------------------------------------
    sources = result.get("sources", [])
    if sources:
        st.subheader("📚 Sources")

        for i, source in enumerate(sources, start=1):
            relevance = source.get("relevance_score", 0.0)
            boosted = source.get("boosted", False)

            # Show relevance as a clean percentage, not a raw float
            relevance_pct = f"{relevance * 100:.1f}%"
            boost_tag = " · 🏷 Boosted" if boosted else ""
            expander_label = (
                f"{i}. {source['title']}  —  Relevance: {relevance_pct}{boost_tag}"
            )

            with st.expander(expander_label):
                col_a, col_b = st.columns([4, 1])

                with col_a:
                    st.markdown(f"**Section:** {source.get('section', '')}")
                    st.markdown(f"**Document:** {source.get('source', '')}")
                    preview = source.get("preview", "")
                    if preview:
                        st.caption(preview)

                with col_b:
                    # Progress bar on 0–1 cosine scale — now correct
                    st.progress(min(max(relevance, 0.0), 1.0))
                    st.caption(relevance_pct)

    st.divider()

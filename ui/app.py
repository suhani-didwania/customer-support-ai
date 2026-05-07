"""
Streamlit chat UI.

Sits on top of the LangGraph multi-agent system: a chat panel for asking
questions, a sidebar for uploading and managing the policy documents that
back the RAG agent, and a small "trace" expander on each answer that shows
which specialist ran and what it retrieved. The trace is helpful for both
debugging and trust - a support agent should be able to see why the assistant
said what it said.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import streamlit as st

from config import PDF_UPLOAD_PATH, OPENAI_API_KEY, SQL_DB_PATH
from utils.vector_store import get_vector_store
from agents.graph import run_query


# --- page setup ---

st.set_page_config(
    page_title="Customer Support AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #fafafa; }

    /* Pull sidebar content up - kill Streamlit's default top padding */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem !important;
    }
    [data-testid="stSidebarUserContent"] {
        padding-top: 0 !important;
    }
    [data-testid="stSidebar"] h2:first-of-type,
    [data-testid="stSidebar"] h3:first-of-type {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    /* Hide the file uploader's "uploaded file" preview row.
       The "Currently indexed" list below is the source of truth. */
    [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
        display: none;
    }

    /* Header */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 1rem;
    }

    /* Description card */
    .info-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 3px solid #6366f1;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin-bottom: 1.5rem;
    }
    .info-card p {
        font-size: 0.9rem;
        color: #4b5563;
        margin: 0.35rem 0;
        line-height: 1.55;
    }
    .info-card p:first-child { margin-top: 0; }
    .info-card p:last-child  { margin-bottom: 0; }
    .info-card em {
        color: #1f2937;
        font-style: italic;
    }

    /* Route badges */
    .route-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .route-sql      { background-color: #dbeafe; color: #1e40af; }
    .route-rag      { background-color: #dcfce7; color: #166534; }
    .route-hybrid   { background-color: #fef3c7; color: #92400e; }
    .route-chitchat { background-color: #f3e8ff; color: #6b21a8; }
</style>
""", unsafe_allow_html=True)


# --- header ---

st.markdown(
    '<p class="main-header">Customer Support AI Assistant</p>',
    unsafe_allow_html=True,
)


# --- sidebar ---

with st.sidebar:
    st.header("Policy Documents")

    # Use a counter as the uploader's key so we can "reset" it after ingest
    # by bumping the counter. This forces Streamlit to mount a fresh uploader.
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_file = st.file_uploader(
        "Upload a policy PDF",
        type=["pdf"],
        key=f"pdf_uploader_{st.session_state.uploader_key}",
        help=(
            "The PDF will be chunked, embedded, and indexed into the vector "
            "store. Long documents can take 30-60 seconds."
        ),
    )

    if uploaded_file is not None:
        try:
            vs = get_vector_store()

            if vs.is_indexed(uploaded_file.name):
                st.info(
                    f"`{uploaded_file.name}` is already indexed. "
                    f"Remove it below if you want to re-index."
                )
                st.session_state.uploader_key += 1
                time.sleep(1.2)
                st.rerun()
            else:
                save_path = Path(PDF_UPLOAD_PATH) / uploaded_file.name
                save_path.parent.mkdir(exist_ok=True, parents=True)

                progress = st.progress(0, text=f"Saving {uploaded_file.name}...")
                save_path.write_bytes(uploaded_file.getvalue())
                progress.progress(20, text="Reading and chunking PDF...")
                progress.progress(40, text="Generating embeddings (30-60s for large docs)...")
                n_chunks = vs.ingest_pdf(str(save_path))
                progress.progress(100, text=f"Indexed {n_chunks} chunks")

                st.session_state.uploader_key += 1
                time.sleep(0.5)
                st.rerun()
        except Exception as e:
            st.error(f"Failed to ingest: {e}")

    # List indexed documents with a remove button next to each
    try:
        vs = get_vector_store()
        indexed = vs.list_documents()
        if indexed:
            st.markdown("**Currently indexed:**")
            for doc in indexed:
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"- `{doc}`")
                with col2:
                    if st.button("Remove", key=f"remove_{doc}", help=f"Remove {doc} from index"):
                        with st.spinner(f"Removing {doc}..."):
                            n_removed = vs.remove_document(doc)
                            file_on_disk = Path(PDF_UPLOAD_PATH) / doc
                            if file_on_disk.exists():
                                file_on_disk.unlink()
                            st.success(f"Removed {n_removed} chunks from {doc}")
                            st.rerun()
        else:
            st.info("No PDFs uploaded yet. Upload them above.")
    except Exception as e:
        st.warning(f"Vector store check failed: {e}")

    st.divider()
    st.header("Sample Questions")
    samples = [
        "Give me an overview of merchant Ema's account and recent activity",
        "What does our agreement say about non-refundable fees?",
        "How many merchants are currently in 'Under Review' status?",
        "Under what conditions can we suspend or terminate a merchant's account?",
        "List all merchants on the Enterprise plan with their monthly volume",
        "Does merchant Ema's account meet our standards under the Stripe terms?",
    ]
    for s in samples:
        if st.button(s, key=f"sample_{hash(s)}", use_container_width=True):
            st.session_state.queued_query = s
            st.rerun()

    st.divider()
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- chat state ---

if "messages" not in st.session_state:
    st.session_state.messages = []
if "queued_query" not in st.session_state:
    st.session_state.queued_query = None


# --- empty state on first load ---

if not st.session_state.messages:
    st.markdown(
        """
        <div class="info-card">
            <p>Ask about a specific merchant by name, business name, or email - the assistant pulls their full profile, transaction history, disputes, and open tickets.</p>
            <p>Ask about a policy in plain English - it searches indexed documents and cites the source.</p>
            <p>For questions that need both, like <em>"is this merchant compliant with our terms?"</em>, it combines the two and gives a reasoned verdict.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- render existing messages ---

ROUTE_LABEL = {
    "sql": "SQL Agent",
    "rag": "RAG Agent",
    "hybrid": "Hybrid (SQL + RAG)",
    "chitchat": "Conversational",
}


def _render_trace(sql_result: str, rag_result: str) -> None:
    """Show what each specialist agent retrieved. Used inside an expander."""
    if sql_result:
        st.markdown("**SQL agent findings:**")
        st.markdown(sql_result[:2000].replace("$", "\\$"))
    if rag_result:
        st.markdown("**RAG agent findings:**")
        st.markdown(rag_result[:2000].replace("$", "\\$"))


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("route"):
            route = msg["route"]
            label = ROUTE_LABEL.get(route, route)
            st.markdown(
                f'<span class="route-badge route-{route}">{label}</span>',
                unsafe_allow_html=True,
            )
        # Escape $ so Streamlit's markdown renderer doesn't treat dollar
        # amounts as LaTeX math delimiters
        st.markdown(msg["content"].replace("$", "\\$"))

        if msg["role"] == "assistant" and (msg.get("sql_result") or msg.get("rag_result")):
            with st.expander("Show agent trace"):
                _render_trace(msg.get("sql_result", ""), msg.get("rag_result", ""))


# --- handle new input ---

prompt = st.chat_input("Ask about customers, tickets, or company policies...")

if st.session_state.queued_query:
    prompt = st.session_state.queued_query
    st.session_state.queued_query = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Routing your question...")

        start = time.time()
        try:
            result = run_query(prompt)
            elapsed = time.time() - start

            route = result.get("route", "unknown")
            label = ROUTE_LABEL.get(route, route)

            placeholder.markdown(
                f'<span class="route-badge route-{route}">{label}</span> '
                f'<span style="color:#6b7280; font-size:0.85rem;">'
                f'(answered in {elapsed:.1f}s)</span>',
                unsafe_allow_html=True,
            )
            st.markdown(result["answer"].replace("$", "\\$"))

            if result.get("sql_result") or result.get("rag_result"):
                with st.expander("Show agent trace"):
                    _render_trace(result.get("sql_result", ""), result.get("rag_result", ""))

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "route": route,
                "sql_result": result.get("sql_result", ""),
                "rag_result": result.get("rag_result", ""),
            })
        except Exception as e:
            err_msg = f"Something went wrong: `{e}`"
            placeholder.markdown(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})
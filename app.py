import streamlit as st

st.set_page_config(
    page_title="IEEE RAS Assistant",
    page_icon="🤖",
    layout="centered",
)


@st.cache_resource(show_spinner="Setting up the knowledge base (first load only)...")
def bootstrap_knowledge_base():
    import ingest

    return ingest.main()


try:
    CHUNK_COUNT = bootstrap_knowledge_base()
    BOOTSTRAP_ERROR = None
except Exception as exc:
    CHUNK_COUNT = 0
    BOOTSTRAP_ERROR = exc

from rag_engine import answer_question  # noqa: E402

SUGGESTED_QUESTIONS = [
    "What is IEEE RAS?"
]

with st.sidebar:
    st.header("🤖 IEEE RAS Assistant")
    st.caption(
        "A RAG-based assistant that answers using an indexed IEEE RAS knowledge base."
    )
    st.markdown("---")
    st.subheader("Try asking")
    for question in SUGGESTED_QUESTIONS:
        if st.button(question, use_container_width=True):
            st.session_state.pending_question = question

    st.markdown("---")
    if CHUNK_COUNT:
        st.caption(f"Knowledge base: {CHUNK_COUNT} indexed passages.")
    st.caption(
        "Answers come only from the indexed documents. When nothing relevant is "
        "found, the assistant says so instead of guessing."
    )
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("IEEE RAS Chapter Assistant")
st.write("Ask about IEEE RAS — membership, events, or general information.")

if BOOTSTRAP_ERROR is not None:
    st.error(f"The knowledge base could not be built: {BOOTSTRAP_ERROR}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_extras(sources, chunks):
    if sources:
        st.caption("📎 Sources: " + ", ".join(sources))
    if chunks:
        with st.expander("Show retrieved context"):
            for chunk in chunks:
                st.markdown(f"**{chunk['source']}** — distance {chunk['distance']:.3f}")
                preview = chunk["text"][:500]
                st.text(preview + ("..." if len(chunk["text"]) > 500 else ""))


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_extras(message.get("sources"), message.get("chunks"))

prompt = st.session_state.pop("pending_question", None)
typed = st.chat_input("Ask a question about IEEE RAS...")
if typed:
    prompt = typed

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            result = answer_question(prompt, chat_history=history, stream=True)

        answer = result["answer"]
        if isinstance(answer, str):
            st.markdown(answer)
            final_text = answer
        else:
            final_text = st.write_stream(answer)

        render_extras(result["sources"], result["chunks"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_text,
            "sources": result["sources"],
            "chunks": result["chunks"],
        }
    )

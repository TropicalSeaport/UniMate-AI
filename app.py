import os
import sys

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np


# =========================
# API Client
# =========================
client = OpenAI(
    api_key=st.secrets["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def clean_text(text):
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    return text


def build_prompt(mode, context, question):
    if mode == "Research Paper Summary":
        task = """
You are summarizing a research paper.
Focus on:
- research background
- problem statement
- core idea
- methodology
- system/model/framework design
- experiments or evaluation
- main contributions
- limitations
- practical value
"""
    elif mode == "Lecture Summary":
        task = """
You are summarizing lecture materials.
Focus on:
- core concepts
- important definitions
- mechanisms
- examples
- what students should remember
"""
    elif mode == "Exam Preparation":
        task = """
You should help the student prepare for exams.
Focus on:
- likely exam points
- key definitions
- formulas
- comparison tables
- possible short-answer questions
- revision notes
"""
    elif mode == "Assignment Explanation":
        task = """
You should help the student understand the assignment.
Focus on:
- task requirements
- deliverables
- grading criteria
- completion steps
- possible mistakes
"""
    else:
        task = """
Answer the user's question clearly based on the retrieved context.
"""

    return f"""
You are UniMate AI, an academic assistant for international students.

You must answer ONLY based on the retrieved document context.
If the context does not contain enough information, say:
"The document does not provide enough information."

Mode:
{mode}

Task:
{task}

Retrieved Document Context:
{context}

User Question:
{question}

Answer Requirements:
- Use the same language as the user's question.
- Be clear and structured.
- Use headings and bullet points when helpful.
- Do not invent information outside the context.
- If this is a research paper, do not call it a lecture.
"""


def hybrid_retrieve(question, chunks, index, embedding_model, bm25, top_k):
    history_questions = []

    for chat in st.session_state.chat_history[-3:]:
        history_questions.append(chat["question"])

    full_query = "\n".join(history_questions + [question])
    full_query = clean_text(full_query)

    question_embedding = embedding_model.encode(
        [full_query],
        show_progress_bar=False
    )
    question_embedding = np.array(question_embedding, dtype=np.float32)

    vector_k = min(top_k * 2, len(chunks))
    _, I = index.search(question_embedding, k=vector_k)

    vector_results = {}
    for rank, idx in enumerate(I[0]):
        vector_results[int(idx)] = 1.0 / (rank + 1)

    tokenized_query = full_query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_top_indices = np.argsort(bm25_scores)[::-1][:vector_k]

    bm25_results = {}
    for rank, idx in enumerate(bm25_top_indices):
        bm25_results[int(idx)] = 1.0 / (rank + 1)

    final_scores = {}

    for idx, score in vector_results.items():
        final_scores[idx] = final_scores.get(idx, 0) + score

    for idx, score in bm25_results.items():
        final_scores[idx] = final_scores.get(idx, 0) + score

    ranked_indices = sorted(
        final_scores.keys(),
        key=lambda x: final_scores[x],
        reverse=True
    )[:top_k]

    retrieved_chunks = [chunks[i] for i in ranked_indices]

    return retrieved_chunks, ranked_indices


def build_research_paper_context(chunks, retrieved_chunks, max_intro_chunks=6):
    intro_chunks = chunks[:max_intro_chunks]
    combined = intro_chunks + retrieved_chunks

    seen = set()
    unique_chunks = []

    for chunk in combined:
        key = chunk[:120]
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)

    return "\n\n".join(unique_chunks)


st.set_page_config(
    page_title="UniMate AI",
    page_icon="🎓",
    layout="wide"
)

st.title("UniMate AI")
st.caption("Hybrid RAG Academic Assistant for International Students")

st.write(
    "Upload lecture slides, assignment briefs, or research papers. "
    "UniMate AI uses Hybrid Retrieval + RAG to answer questions with source context."
)


if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "bm25" not in st.session_state:
    st.session_state.bm25 = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.sidebar.title("Settings")

mode = st.sidebar.selectbox(
    "Choose Mode",
    [
        "Research Paper Summary",
        "Lecture Summary",
        "Exam Preparation",
        "Assignment Explanation",
        "General QA"
    ]
)

top_k = st.sidebar.slider(
    "Number of retrieved chunks",
    min_value=1,
    max_value=8,
    value=5
)

if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.sidebar.success("Chat history cleared.")


uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)

question = st.text_input(
    "Ask a question about your document:",
    placeholder="例如：用中文总结这篇论文的研究背景、核心贡献、方法和实验结果"
)


if uploaded_file:
    pdf_text = ""

    reader = PdfReader(uploaded_file)

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pdf_text += clean_text(text) + "\n"

    pdf_text = clean_text(pdf_text)

    st.success("PDF uploaded and parsed successfully!")

    with st.expander("Preview document text"):
        st.write(pdf_text[:3000])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    )

    chunks = text_splitter.split_text(pdf_text)

    chunks = [
        clean_text(chunk)
        for chunk in chunks
        if clean_text(chunk).strip()
    ]

    st.info(f"Document split into {len(chunks)} chunks.")

    if len(chunks) == 0:
        st.error("No valid text was extracted from the PDF.")
    else:
        with st.spinner("Building Hybrid Retrieval Index..."):
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            chunk_embeddings = embedding_model.encode(
                chunks,
                show_progress_bar=False
            )

            chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)

            dimension = chunk_embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(chunk_embeddings)

            tokenized_chunks = [
                chunk.lower().split()
                for chunk in chunks
            ]

            bm25 = BM25Okapi(tokenized_chunks)

            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.embedding_model = embedding_model
            st.session_state.bm25 = bm25

        st.success("Hybrid retrieval index built successfully!")


if st.button("Ask AI"):
    if not uploaded_file:
        st.warning("Please upload a PDF first.")

    elif not question:
        st.warning("Please enter a question.")

    elif st.session_state.index is None:
        st.warning("Index is not ready yet.")

    else:
        safe_question = clean_text(question)

        retrieved_chunks, source_indices = hybrid_retrieve(
            question=safe_question,
            chunks=st.session_state.chunks,
            index=st.session_state.index,
            embedding_model=st.session_state.embedding_model,
            bm25=st.session_state.bm25,
            top_k=top_k
        )

        if mode == "Research Paper Summary":
            context = build_research_paper_context(
                chunks=st.session_state.chunks,
                retrieved_chunks=retrieved_chunks,
                max_intro_chunks=6
            )
        else:
            context = "\n\n".join(retrieved_chunks)

        context = clean_text(context)

        prompt = build_prompt(mode, context, safe_question)

        with st.spinner("Generating answer..."):
            response = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

        answer = response.choices[0].message.content

        st.session_state.chat_history.append(
            {
                "mode": mode,
                "question": safe_question,
                "answer": answer,
                "sources": retrieved_chunks,
                "source_indices": source_indices
            }
        )

        st.subheader("AI Answer")
        st.write(answer)

        with st.expander("Retrieved Sources"):
            for idx, chunk in enumerate(retrieved_chunks, start=1):
                st.markdown(f"### Source Chunk {idx}")
                st.write(chunk)


if st.session_state.chat_history:
    st.divider()
    st.subheader("Chat History")

    for i, chat in enumerate(reversed(st.session_state.chat_history), start=1):
        with st.expander(f"Conversation {i} | Mode: {chat['mode']}"):
            st.markdown("**Question**")
            st.write(chat["question"])

            st.markdown("**Answer**")
            st.write(chat["answer"])

            st.markdown("**Retrieved Sources**")
            for idx, source in enumerate(chat["sources"], start=1):
                st.markdown(f"Source {idx}")
                st.write(source)

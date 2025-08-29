import importlib
import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# -------------------------
# 1. Load embedding model (Granite Embeddings)
# -------------------------
embedder = SentenceTransformer("ibm-granite/granite-embedding-english-r2")

# -------------------------
# 2. Load generation model (Granite Instruct)
# -------------------------
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-2b-instruct")

# Check if accelerate is installed
if importlib.util.find_spec("accelerate") is not None:
    # Use accelerate auto device placement
    gen_model = AutoModelForCausalLM.from_pretrained(
        "ibm-granite/granite-3.3-2b-instruct",
        device_map="auto",
        torch_dtype="auto"
    )
    generator = pipeline(
        "text-generation",
        model=gen_model,
        tokenizer=tokenizer
    )
else:
    # Fallback: CPU only (no accelerate installed)
    gen_model = AutoModelForCausalLM.from_pretrained(
        "ibm-granite/granite-3.3-2b-instruct"
    )
    generator = pipeline(
        "text-generation",
        model=gen_model,
        tokenizer=tokenizer,
        device=-1  # force CPU
    )

# -------------------------
# 3. PDF Text Extraction
# -------------------------
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join(page.get_text("text") for page in doc)

# -------------------------
# 4. Chunking
# -------------------------
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# -------------------------
# 5. FAISS Index Build
# -------------------------
def build_faiss(chunks):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

# -------------------------
# 6. Query Granite
# -------------------------
def query_granite(question, retrieved_chunks):
    prompt = (
        "Answer the following question based strictly on the provided context.\n\n"
        "Context:\n" + "\n\n".join(f"- {c}" for c in retrieved_chunks)
        + f"\n\nQuestion: {question}\n\nAnswer:"
    )
    out = generator(prompt, max_new_tokens=200, do_sample=False)
    return out[0]["generated_text"].split("Answer:")[-1].strip()

# -------------------------
# 7. Streamlit UI
# -------------------------
st.set_page_config(page_title="StudyMate (Granite Hugging Face)", layout="wide")
st.title("📘 StudyMate – PDF Q&A with IBM Granite 🚀")

uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    st.success(f"✅ Processed {len(uploaded_files)} files – total {len(all_chunks)} chunks.")

    faiss_index, vectors = build_faiss(all_chunks)

    question = st.text_input("Ask a question:")
    if question:
        q_emb = embedder.encode([question], convert_to_numpy=True)
        _, I = faiss_index.search(q_emb, k=3)
        retrieved = [all_chunks[idx] for idx in I[0]]

        answer = query_granite(question, retrieved)

        st.subheader("📌 Answer")
        st.write(answer)

        with st.expander("📖 Source Context"):
            for chunk in retrieved:
                st.markdown(f"> {chunk}")


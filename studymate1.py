import streamlit as st
import fitz  # PyMuPDF
import re
from collections import Counter
import math

st.set_page_config(page_title="StudyMate (Offline)", layout="wide")
st.title("📘 StudyMate – Offline PDF Q&A System 🚀")

st.info("🌐 **Offline Mode**: This version works without internet connection using keyword-based search!")

# -------------------------
# Offline Text Processing Functions
# -------------------------

def extract_text_from_pdf(file):
    """Extract text from PDF"""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + "\n"
    return full_text

def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
    return chunks

def preprocess_text(text):
    """Clean and preprocess text for searching"""
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace and special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def calculate_similarity(query_words, chunk_words):
    """Calculate similarity between query and chunk using TF-IDF-like scoring"""
    # Count word frequencies in chunk
    chunk_counter = Counter(chunk_words)
    total_chunk_words = len(chunk_words)
    
    score = 0
    for word in query_words:
        if word in chunk_counter:
            # TF (term frequency)
            tf = chunk_counter[word] / total_chunk_words
            # Simple scoring: give higher weight to exact matches
            score += tf * 2
            
            # Bonus for multiple occurrences of the same word
            if chunk_counter[word] > 1:
                score += 0.5
    
    return score

def search_chunks(query, chunks, top_k=3):
    """Search for most relevant chunks using keyword matching"""
    query_processed = preprocess_text(query)
    query_words = query_processed.split()
    
    chunk_scores = []
    
    for i, chunk in enumerate(chunks):
        chunk_processed = preprocess_text(chunk)
        chunk_words = chunk_processed.split()
        
        similarity = calculate_similarity(query_words, chunk_words)
        
        # Bonus for exact phrase matches
        if query_processed in chunk_processed:
            similarity += 1.0
        
        chunk_scores.append((i, similarity, chunk))
    
    # Sort by similarity score (descending)
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k chunks
    return [(chunk, score) for _, score, chunk in chunk_scores[:top_k]]

def generate_simple_answer(query, relevant_chunks):
    """Generate a simple answer based on relevant chunks"""
    if not relevant_chunks:
        return "I couldn't find relevant information in the uploaded documents."
    
    # Combine the most relevant chunks
    combined_text = "\n\n".join([chunk for chunk, _ in relevant_chunks])
    
    # Simple answer generation based on query type
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['what', 'define', 'definition']):
        answer_prefix = "Based on the document, here's what I found:\n\n"
    elif any(word in query_lower for word in ['how', 'explain']):
        answer_prefix = "Here's an explanation from the document:\n\n"
    elif any(word in query_lower for word in ['why', 'reason']):
        answer_prefix = "The document provides this information:\n\n"
    elif any(word in query_lower for word in ['when', 'time']):
        answer_prefix = "According to the document:\n\n"
    else:
        answer_prefix = "I found this relevant information:\n\n"
    
    # Truncate if too long
    if len(combined_text) > 800:
        combined_text = combined_text[:800] + "..."
    
    return answer_prefix + combined_text

# -------------------------
# Streamlit Interface
# -------------------------

# File upload
uploaded_files = st.file_uploader(
    "Upload PDF(s)", 
    type="pdf", 
    accept_multiple_files=True,
    help="Upload one or more PDF files to search through"
)

if uploaded_files:
    # Process PDFs
    with st.spinner("📄 Processing PDFs..."):
        all_chunks = []
        pdf_info = []
        
        for i, file in enumerate(uploaded_files):
            try:
                text = extract_text_from_pdf(file)
                chunks = chunk_text(text)
                all_chunks.extend(chunks)
                pdf_info.append({
                    'name': file.name,
                    'chunks': len(chunks),
                    'characters': len(text)
                })
            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {e}")
    
    # Show processing results
    st.success(f"✅ Processed {len(uploaded_files)} PDF(s)")
    
    # PDF info expander
    with st.expander("📊 Document Information"):
        for info in pdf_info:
            st.write(f"**{info['name']}**: {info['chunks']} chunks, {info['characters']:,} characters")
        st.write(f"**Total chunks**: {len(all_chunks)}")
    
    # Search interface
    st.write("---")
    query = st.text_input(
        "🔍 Ask a question about your documents:",
        placeholder="e.g., What is the main topic? How does this work? Define key terms..."
    )
    
    if query:
        with st.spinner("🔍 Searching documents..."):
            relevant_chunks = search_chunks(query, all_chunks, top_k=3)
            answer = generate_simple_answer(query, relevant_chunks)
        
        # Display results
        st.subheader("📌 Answer")
        st.write(answer)
        
        # Show search results
        with st.expander(f"🔍 Search Results ({len(relevant_chunks)} chunks found)"):
            for i, (chunk, score) in enumerate(relevant_chunks, 1):
                st.write(f"**Result {i}** (Score: {score:.2f})")
                st.markdown(f"> {chunk[:300]}{'...' if len(chunk) > 300 else ''}")
                st.write("---")
        
        # Search tips
        with st.expander("💡 Search Tips"):
            st.write("""
            **This offline system works by:**
            - 🔍 Keyword matching and similarity scoring
            - 📊 Finding chunks with the highest relevance scores
            - 🎯 Exact phrase matching gets bonus points
            
            **For better results:**
            - Use specific keywords from your document
            - Try different phrasings of your question
            - Use key terms that likely appear in the text
            """)

else:
    # Instructions
    st.write("## 🚀 How to Use")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### ✅ Features")
        st.write("- 📄 Extract text from PDFs")
        st.write("- 🔍 Keyword-based search")
        st.write("- 🎯 Relevance scoring")
        st.write("- 🌐 **100% Offline** - No internet needed!")
    
    with col2:
        st.write("### 📋 Instructions")
        st.write("1. Upload one or more PDF files")
        st.write("2. Wait for processing to complete") 
        st.write("3. Ask questions about your documents")
        st.write("4. Get answers with source excerpts")

    st.write("---")
    st.info("💡 **Note**: This offline version uses keyword matching instead of AI models. While not as sophisticated as AI-powered search, it works reliably without internet connection!")
import streamlit as st
from pypdf import PdfReader
from model import add_to_vector_db, answer_question

st.set_page_config(page_title="StudyMate", layout="wide")

st.title("📘 StudyMate – AI PDF Q&A")

# ---------------- SIDEBAR ---------- ------
st.sidebar.header("Upload PDF")

pdf_file = st.sidebar.file_uploader("Choose your PDF", type=["pdf"])

if pdf_file:
    st.sidebar.success("PDF uploaded!")

    # Read PDF text
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    st.sidebar.info("Processing PDF...")
    
    add_to_vector_db(text)

    st.sidebar.success("PDF processed!")


# ---------------- MAIN AREA ----------------
st.subheader("Ask a Question 👇")

question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please type a question!")
    else:
        with st.spinner("Thinking..."):
            answer = answer_question(question)
        st.write("### Answer:")
        st.write(answer)

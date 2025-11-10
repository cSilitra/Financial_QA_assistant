import streamlit as st
import queue
import threading
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from main import run_master_agent, crate_vector_store
from utils import delete_vector_store

# set browser tab Title and icon
st.set_page_config(page_title="Ai chat", page_icon="🤖")

# set page title and description
st.title("Company report AI Assistant 🤖")
st.caption("Discover information from uploaded PDF.")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])

# Create two columns for the buttons
col1, col2 = st.columns(2)

# First button in first column
with col1:
    if st.button("Crear vector store"):
        delete_vector_store()
        st.success("Vector store cleared successfully!")

# Second button in second column
with col2:
    if st.button("Upload PDF to vector store"):
        if uploaded_file is not None:
            crate_vector_store(uploaded_file)
            st.success("File uploaded successfully!")
        else:
            st.error("Please upload a PDF file.")


# Text area for user query
query = st.text_area(
    "Describe your query:", value='', height=200
)

# generate response
if st.button("Generate Response") and query:
    try:
        result = run_master_agent(query)
        st.subheader("📊 Final response")
        st.markdown(result)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
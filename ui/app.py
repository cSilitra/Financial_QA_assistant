import streamlit as st
import queue
import threading
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from main import run_master_agent, crate_vector_store, run_gemini_file_search, run_base_llm, run_base_llm
from utils import delete_vector_store

# set browser tab Title and icon
st.set_page_config(page_title="Ai chat", page_icon="🤖")
st.set_page_config(layout="wide")

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
if st.button("Generate Response"):
    if not query:
        st.error("Please enter a query.")
    else:
        try:
            st.subheader("📊 Final response")
            
            # Dictionary to store results from both threads
            results = {'base': None, 'geminiRAG': None, 'base_error': None, 'geminiRAG_error': None,'llm':None, 'llm_error':None}
            
            # Function to run base RAG agent in a thread
            def run_base_agent():
                try:
                    results['base'] = run_master_agent(query)
                except Exception as e:
                    results['base_error'] = str(e)
            
            # Function to run Gemini search in a thread
            def run_gemini_search():
                try:
                    results['geminiRAG'] = run_gemini_file_search(query)
                except Exception as e:
                    results['geminiRAG_error'] = str(e)

          # Function to run basic llm search in a thread
            def run_basic_llm_search():
                try:
                    results['llm'] = run_base_llm(query)
                except Exception as e:
                    results['llm_error'] = str(e)
            
            # Create and start both threads
            thread_base = threading.Thread(target=run_base_agent)
            thread_gemini = threading.Thread(target=run_gemini_search)
            thread_llm = threading.Thread(target=run_basic_llm_search)
            
            thread_base.start()
            thread_gemini.start()
            thread_llm.start()
            
            # Wait for both threads to complete
            thread_base.join()
            thread_gemini.join()
            thread_llm.join()
            
            # Display results from both sources
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Base RAG (gpt-5-mini) Response:**")
                if results['base_error']:
                    st.error(f"Error: {results['base_error']}")
                else:
                    st.markdown(results['base'])
            
            with col2:
                st.write("**Gemini-2.5-flash response:**")
                if results['geminiRAG_error']:
                    st.error(f"Error: {results['geminiRAG_error']}")
                else:
                    st.markdown(results['geminiRAG'])
            with col3:
                st.write("**LLM(gpt-5-mini) Response:**")
                if results['llm_error']:
                    st.error(f"Error: {results['llm_error']}")
                else:
                    st.markdown(results['llm'])
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
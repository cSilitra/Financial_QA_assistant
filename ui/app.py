import streamlit as st
import queue
import threading
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from main import run_master_agent

st.set_page_config(page_title="Ai chat", page_icon="🤖")

st.title("Company report AI Assistant 🤖")
st.caption("Discover information from uploaded PDF.")

# Text area for user query
query = st.text_area(
    "Describe your query:", value='', height=200
)

if st.button("Generate Response") and query:
   result = run_master_agent(query)
   st.subheader("📊 Final response")
   st.markdown(result)
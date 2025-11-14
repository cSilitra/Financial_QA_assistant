import os
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types
import time

_ = load_dotenv(find_dotenv())
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)
def run_gemini_search_rag(query):
    store = client.file_search_stores.create()

    upload_op = client.file_search_stores.upload_to_file_search_store(
        file_search_store_name=store.name,
        file='googleQ32025.pdf'
    )

    while not upload_op.done:
        time.sleep(5)
        upload_op = client.operations.get(operation=upload_op)

    # Use the file search store as a tool in your generation call
    # contents='Summarize the key findings from the uploaded document "googleQ32025.pdf"',
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=query,
        config=types.GenerateContentConfig(
            tools=[types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[store.name]
                )
            )]
        )
    )
    return response.text
   

# Support your response with links to the grounding sources.
#grounding = response.candidates[0].grounding_metadata
#sources = {c.retrieved_context.title for c in grounding.grounding_chunks}
#print('Sources:', *sources)
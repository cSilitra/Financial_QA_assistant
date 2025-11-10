# Financial QA Assistant 📊

An intelligent document analysis system that leverages AI to provide insights from financial documents. This assistant processes PDF documents, stores them in a vector database, and answers questions based on the uploaded content.

## Features 🚀

- **PDF Document Processing**: Upload and analyze multiple financial documents
- **Intelligent Question Answering**: Get accurate responses based on uploaded documents
- **Persistent Knowledge Base**: Documents are stored in a local vector store for quick retrieval
- **Interactive UI**: User-friendly Streamlit interface for document management and queries
- **Incremental Learning**: Add new documents without losing existing knowledge
- **Easy Document Management**: Clear or update the vector store as needed

## Getting Started 🏁

### Prerequisites

- Python 3.8+
- Required packages (install via `pip`):
  - streamlit
  - langchain
  - chromadb
  - unstructured
  - python-dotenv

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/cSilitra/Financial_QA_assistant.git
   cd Financial_QA_assistant
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Start the application using either:
```bash
streamlit run ui/app.py
```
or
```bash
npm start
```

## Usage Guide 📖

1. **Upload Documents**
   - Click the "Upload PDF" button to add financial documents
   - Multiple documents can be uploaded and stored

2. **Manage Vector Store**
   - Use "Crear vector store" to clear the existing knowledge base
   - Use "Upload PDF to vector store" to add new documents

3. **Ask Questions**
   - Enter your query in the text area
   - Click "Generate Response" to get AI-powered insights
   - Responses are provided in markdown format for better readability

## Technical Details 🛠️

- Uses LangChain for document processing and RAG (Retrieval Augmented Generation)
- Implements ChromaDB as the vector store for document embeddings
- Supports both local PDF processing and Unstructured API integration
- Features automatic chunk splitting for optimal document processing

## Contributing 🤝

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/cSilitra/Financial_QA_assistant/issues).

## License 📝

[MIT License](LICENSE)

---
Built with ❤️ by [cSilitra](https://github.com/cSilitra)
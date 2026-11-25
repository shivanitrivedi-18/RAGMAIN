# RAGMAN – Retrieval-Augmented Generation

## Overview

RAGMAN is a Retrieval-Augmented Generation (RAG) project that allows users to interact with documents using natural language questions.

The project combines document retrieval with a Large Language Model (LLM) to generate context-aware responses based on the information available in the uploaded documents.

## Features

- PDF document loading and processing
- Text extraction and document chunking
- Semantic search using vector embeddings
- Vector storage using ChromaDB
- Retrieval of relevant document content
- Context-aware question answering using an LLM

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | RAG pipeline and component integration |
| PyPDFLoader | PDF document loading |
| RecursiveCharacterTextSplitter | Splitting documents into chunks |
| HuggingFace Embeddings | Converting text into vector representations |
| ChromaDB | Vector database for storing and retrieving embeddings |
| Large Language Model | Generating responses from retrieved context |

## Project Workflow

1. Load the PDF document.
2. Extract text from the document.
3. Split the extracted text into smaller chunks.
4. Generate embeddings for the text chunks.
5. Store the embeddings in ChromaDB.
6. Convert the user's question into an embedding.
7. Retrieve relevant document chunks.
8. Pass the retrieved context and question to the LLM.
9. Generate a context-aware answer.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/RAGMAN.git
cd RAGMAN
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows:**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project directory and add the API key required by your LLM provider.

```env
GROQ_API_KEY=your_api_key_here
```

Keep your API key private. Never upload your `.env` file to GitHub.

### 5. Run the project

```bash
python main.py
```

Replace `main.py` with your actual Python entry-point filename if it is different.

## Project Structure

```text
RAGMAN/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
│
└── Other project files and folders
```

The project structure may vary depending on the implementation.

## Applications

- Document-based question answering
- Research paper analysis
- Educational document assistants
- Knowledge retrieval from PDFs
- Information extraction from large documents

## Future Improvements

- Support for multiple document formats
- Web-based user interface
- Conversation history
- Source citations in generated answers
- Improved retrieval accuracy

## Author

**Shivani Trivedi**

B.Tech Computer Science – Data Science & Artificial Intelligence

Shri Ramswaroop Memorial University, Lucknow

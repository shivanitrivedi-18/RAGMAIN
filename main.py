import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


PROJECT_DIR = Path(__file__).resolve().parent
PDF_PATH = PROJECT_DIR / "Programming.pdf"
COLLECTION_NAME = "programming_knowledge"

load_dotenv(dotenv_path=PROJECT_DIR / ".env")


def build_retriever():
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"Reference PDF not found: {PDF_PATH}")

    documents = PyPDFLoader(str(PDF_PATH)).load()
    if not documents:
        raise ValueError(f"No readable pages found in {PDF_PATH}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError(f"No text could be extracted from {PDF_PATH}")

    print(f"Loaded {len(documents)} pages and created {len(chunks)} chunks.")
    print("Creating embeddings and indexing the PDF...")

    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    vectorstore.add_documents(chunks)
    print("PDF indexing complete.")
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def build_rag_chain():
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is missing. Add NVIDIA_API_KEY=<your key> to the .env file."
        )

    llm = ChatNVIDIA(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        nvidia_api_key=api_key,
        temperature=0.2,
        max_completion_tokens=2500,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a programming tutor. Answer using only the supplied "
                "document context. If the context does not contain the answer, "
                "say you do not know.\n\nContext:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def query_rag(question: str, retriever, rag_chain):
    retrieved_docs = retriever.invoke(question)
    if not retrieved_docs:
        raise RuntimeError("The PDF retriever returned no relevant text.")

    print(f"\nQuestion: {question}")
    print(f"Retrieved {len(retrieved_docs)} relevant chunk(s):")
    for index, document in enumerate(retrieved_docs, start=1):
        page_number = document.metadata.get("page")
        if isinstance(page_number, int):
            page_number += 1
        preview = document.page_content.strip().replace("\n", " ")[:150]
        print(f" [{index}] Page {page_number}: {preview}...")

    context = "\n\n".join(document.page_content for document in retrieved_docs)
    print("\nGenerating answer with NVIDIA...")
    answer = rag_chain.invoke({"context": context, "question": question})
    print(f"\nAnswer:\n{answer}")
    return answer


if __name__ == "__main__":
    retriever = build_retriever()
    rag_chain = build_rag_chain()
    query_rag(
        "According to the document, what are the main steps in a systematic debugging process?",
        retriever,
        rag_chain,
    )

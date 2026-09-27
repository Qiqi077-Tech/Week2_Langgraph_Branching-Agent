"""Load the Apple 10-K PDF into Pinecone, one chunk per page."""

import os
import re

from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader

from vectorstore import EMBEDDING_DIMENSION, INDEX_NAME, get_vector_store

PDF_PATH = "_10-K-2025-As-Filed.pdf"
# Footer on each page, e.g. "Apple Inc. | 2025 Form 10-K | 26" (differs from the PDF page).
FOOTER_PATTERN = re.compile(r"Apple Inc\. \| 2025 Form 10-K \| (\d+)")


def load_pages(path: str) -> list[Document]:
    reader = PdfReader(path)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            metadata = {"source": path, "page": number}
            footer = FOOTER_PATTERN.search(text)
            if footer:
                metadata["printed_page"] = int(footer.group(1))
            pages.append(Document(page_content=text, metadata=metadata))
    return pages


def ensure_index() -> None:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def main() -> None:
    pages = load_pages(PDF_PATH)
    print(f"Loaded {len(pages)} pages from {PDF_PATH}")
    ensure_index()
    # Page number as the id makes re-running idempotent (upserts over existing pages).
    ids = [f"page-{doc.metadata['page']}" for doc in pages]
    get_vector_store().add_documents(pages, ids=ids)
    print(f"Upserted {len(pages)} pages into Pinecone index '{INDEX_NAME}'")


if __name__ == "__main__":
    main()
